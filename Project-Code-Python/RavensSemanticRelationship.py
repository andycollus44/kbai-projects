from abc import ABCMeta, abstractproperty, abstractmethod
from operator import attrgetter

import numpy as np
from PIL import Image, ImageDraw

from RavensShape import RavensRelativeSizeRanker, RavensShapeExtractor, RavensShapeMatcher

# Keep singleton instances available to all semantic relationships
_extractor = RavensShapeExtractor()
_matcher = RavensShapeMatcher()


class SemanticRelationship:
    __metaclass__ = ABCMeta

    @abstractproperty
    def name(self):
        pass

    @abstractmethod
    def generate(self, ravens_matrix, axis):
        """
        Generates semantic relationship given the Raven's matrix of images and an axis,
        and applies this relationship to generate an expected result.

        :param ravens_matrix: The matrix of images that constitutes the Raven's problem.
        :type ravens_matrix: list[list[PIL.Image.Image]]
        :param axis: The axis from which to generate the relationship.
        :type axis: int
        :return: The expected result obtained after generating the relationship.
                 This result could be anything from another image to a single number.
        """
        pass

    @abstractmethod
    def test(self, expected, ravens_matrix, answers, axis):
        """
        Tests the generated expected result against the available answers to the problem.

        :param expected: The expected result generated from a call to `generate()`.
        :param ravens_matrix: The matrix of images that constitutes the Raven's problem.
        :type ravens_matrix: list[list[PIL.Image.Image]]
        :param answers: The set of available answers.
        :type answers: list[PIL.Image.Image]
        param axis: The axis from which the relationship was generated.
        :type axis: int
        :return: The correct answer or None, if no answer matches the expected results.
        :rtype: int
        """
        pass

    def is_valid(self, axis):
        """
        Determines whether this relationship is valid for the given axis.
        By default all axes are valid and underlying relationships should override this method accordingly.

        :param axis: The axis this relationships is being evaluated against.
        :return: True if this relationship is valid for the given axis, False otherwise.
        """
        return True

    def __repr__(self):
        return self.name


class SemanticRelationship3x3(SemanticRelationship):
    __metaclass__ = ABCMeta

    def _select_first_image(self, matrix, axis):
        # The 'G' image for row-wise, the 'C' image for column-wise or the 'A' image for diagonal-wise
        if axis == 0:
            return matrix[2][0]
        elif axis == 1:
            return matrix[0][2]
        else:
            return matrix[0][0]

    def _select_second_image(self, matrix, axis):
        # The 'H' image for row-wise, the 'F' image for column-wise or the 'E' image for diagonal-wise
        if axis == 0:
            return matrix[2][1]
        elif axis == 1:
            return matrix[1][2]
        else:
            return matrix[1][1]


class AddKeepDelete2x2(SemanticRelationship):
    """
    Generates a semantic relationship of shapes added, deleted or kept between frames for 2x2 matrices.
    """
    ADD = 'ADD'
    DELETE = 'DELETE'
    KEEP = 'KEEP'

    @property
    def name(self):
        return 'Add-Keep-Delete'

    def generate(self, ravens_matrix, axis):
        # Step 1: Get all the shapes for 'A', 'B' and 'C'
        shapes_a = _extractor.apply(ravens_matrix[0][0])
        shapes_b = _extractor.apply(ravens_matrix[0][1])
        shapes_c = _extractor.apply(ravens_matrix[1][0])

        # Step 2: Find the matching shapes between 'A' and 'B' and 'A' and 'C', and consolidate labels
        self._consolidate_labels(shapes_a, shapes_b, shapes_c)

        # Step 3: Build relationships between 'A' to 'B' for row-wise or 'A' to 'C' for column-wise
        relationships = self._build_relationships(shapes_a, shapes_b if axis == 0 else shapes_c)

        # Step 4: Apply each relationship to the last image, 'C' for row-sie or 'B' for column-wise
        expected_shapes = self._apply_relationships(relationships, shapes_c if axis == 0 else shapes_b)

        # Step 5: Reconstruct the expected result back into an image, this will help consolidating shapes
        reconstructed = self._reconstruct(expected_shapes)

        return reconstructed

    def test(self, expected, ravens_matrix, answers, axis):
        # Re-apply the extractor to the expected result, this is important to handle cases where the
        # expected result ends up being a single filled shape, which though it was originally composed
        # by many shapes, visually, it is now a single one, so the answer will only have one too!
        result = _extractor.apply(expected)

        # Handle empty images as the expected result
        if len(result) == 0:
            return self._test_empty_image(answers)
        else:
            # Do a shape-by-shape comparison for each answer
            return self._test_shape_by_shape(result, answers)

    def _consolidate_labels(self, shapes_a, shapes_b, shapes_c):
        for index, b in enumerate(shapes_b):
            match = self._best_shape_match(b, shapes_a)

            if not match:
                continue

            b.label = match.label

        for c in shapes_c:
            match = self._best_shape_match(c, shapes_a)

            if not match:
                continue

            c.label = match.label

    def _build_relationships(self, shapes_a, other_shapes):
        # Builds semantic relationships between the shapes in 'A' and the other given shapes
        # Since the labels were consolidated, they can be used to identify shapes deleted, added or kept between frames
        a_labels = set([a.label for a in shapes_a])
        other_labels = set([o.label for o in other_shapes])

        # A dictionary that will contain the relationships in the form label => (relationship, shape)
        relationships = {}

        # All shapes kept or deleted from 'A'
        for shape in shapes_a:
            if shape.label in other_labels:
                relationships[shape.label] = (self.KEEP, shape)
            else:
                relationships[shape.label] = (self.DELETE, shape)

        # All shapes added from the other frame
        for shape in other_shapes:
            if shape.label not in a_labels:
                relationships[shape.label] = (self.ADD, shape)

        return relationships

    def _apply_relationships(self, relationships, shapes):
        # Applies the relationships to the given shapes to generate an expected result as a list of shapes

        # Add any unchanged shapes that that do not have a relationship since they could be either
        # additions to the image which are not related to the problem or transformed shapes (e.g.
        # from square to triangle) which do not have a match, but based on addition and deletion, this
        # transformation will be implicitly performed
        expected = [shape for shape in shapes
                    if shape.label not in relationships.keys() or relationships[shape.label][0] == self.KEEP]

        # Now, add in any other shapes whose relationships is ADD and were not already present
        expected = expected + [shape for relationship, shape in relationships.itervalues() if relationship == self.ADD]

        return expected

    def _reconstruct(self, shapes):
        reconstructed = Image.new('L', (184, 184), 255)
        draw = ImageDraw.Draw(reconstructed)

        # Handle reconstructing blank images
        if len(shapes) == 0:
            return reconstructed

        # Run the ranker to assign new size ranks to the reconstructed image
        # This is very important because previous ranks are not valid anymore,
        # e.g. a shape that was the smallest (rank 0), might not be it anymore
        sizes = RavensRelativeSizeRanker.apply(shapes)
        for shape in shapes:
            shape.size_rank = sizes[shape.label]

        # To make sure smaller shapes, that are inside any filled ones, are not masked,
        # sort the shapes based on the size rank in decreasing, so that smaller shapes are drawn last
        shapes = sorted(shapes, key=attrgetter('size_rank'), reverse=True)

        # Reconstruct the image by drawing all the shapes
        for shape in shapes:
            draw.polygon(shape.contour.flatten().tolist(), fill=0 if shape.filled else 255, outline=0)

        return reconstructed

    def _test_empty_image(self, answers):
        for index, answer in enumerate(answers):
            # Look for answer that also has zero shapes
            answer_shapes = _extractor.apply(answer)

            if len(answer_shapes) == 0:
                return index + 1

        return None

    def _test_shape_by_shape(self, expected_shapes, answers):
        for index, answer in enumerate(answers):
            all_match = True
            answer_shapes = _extractor.apply(answer)

            # Make sure the answer has the same shapes as the expected result
            if len(answer_shapes) != len(expected_shapes):
                continue

            # For each of the shapes in the expected result,
            # make sure the answer under test contains it too
            for expected_shape in expected_shapes:
                match = self._best_shape_match(expected_shape, answer_shapes)

                if not match:
                    # This answer does not comply with all the expected shapes
                    all_match = False
                    break

            if all_match:
                # We have found an answer that complies with all the expected shapes!
                return index + 1

        return None

    def _best_shape_match(self, shape, other_shapes):
        # Finds the shape that best matches the given shape out of the list of the other shapes using a full matching
        similarities = [_matcher.apply(shape, other_shape, match=_matcher.MATCH_ALL) for other_shape in other_shapes]
        match = np.argmax(similarities)

        return other_shapes[match] if similarities[match] > 0.9 else None


class SidesArithmetic(SemanticRelationship):
    """
    Generates a semantic relationship of arithmetic between frames based on shapes' sides for 2x2 matrices.
    This semantic relationship is only going to be valid for problems where there are geometric figures.
    """
    @property
    def name(self):
        return 'Sides Arithmetic'

    def generate(self, ravens_matrix, axis):
        # Get the shapes for 'A' and 'B' (row-wise) or 'C' (column-wise)
        shapes_a = _extractor.apply(ravens_matrix[0][0])
        other_shapes = _extractor.apply(ravens_matrix[0][1]) if axis == 0 else _extractor.apply(ravens_matrix[1][0])

        # For each list of shapes, count the number of sides of each shape
        sides_a = sum([s.sides for s in shapes_a])
        other_sides = sum([s.sides for s in other_shapes])

        # The expected result is the difference between sides
        return sides_a - other_sides

    def test(self, expected, ravens_matrix, answers, axis):
        # Get the shapes for 'C' (row-wise) or 'B' (column-wise), and the count of their sides
        shapes = _extractor.apply(ravens_matrix[1][0]) if axis == 0 else _extractor.apply(ravens_matrix[0][1])
        sides = sum([s.sides for s in shapes])

        # For each answer, extract the shapes and the count of their sides
        for index, answer in enumerate(answers):
            answer_shapes = _extractor.apply(answer)
            answer_sides = sum([s.sides for s in answer_shapes])

            # If the difference between the answer and the image's sides equals the expected,
            # then we have found an answer
            if sides - answer_sides == expected:
                return index + 1

        return None


class ShapeFillPointsSystem3x3(SemanticRelationship3x3):
    """
    Generates a semantic relationship of a points system based on the 'filled' attribute of shapes for a 3x3 problem.
    """
    # Point system based on filled/non-filled attributes of shapes
    # Note that the relationship would still hold if we had chosen other values,
    # as long as, these values are double of each other, e.g. 2 and 4.
    _POINTS_FILLED = 1
    _POINTS_EMPTY = 2

    @property
    def name(self):
        return 'Shape Fill Points System'

    def generate(self, ravens_matrix, axis):
        # Extract the shapes for the first and second image
        shapes_1 = _extractor.apply(self._select_first_image(ravens_matrix, axis))
        shapes_2 = _extractor.apply(self._select_second_image(ravens_matrix, axis))

        # Now count the points for each image based on the system
        points_1 = self._count_points(shapes_1)
        points_2 = self._count_points(shapes_2)

        # The expected result is the difference between points
        return points_1 - points_2

    def test(self, expected, ravens_matrix, answers, axis):
        # Get the shapes for the second image and count the points based on the same system
        shapes = _extractor.apply(self._select_second_image(ravens_matrix, axis))
        points = self._count_points(shapes)

        # For each answer, extract the shapes and count their points
        for index, answer in enumerate(answers):
            answer_shapes = _extractor.apply(answer)
            answer_points = self._count_points(answer_shapes)

            # If the difference between the image's and the answer's points equals the expected,
            # then we have found an answer
            if points - answer_points == expected:
                return index + 1

        return None

    def _count_points(self, shapes):
        if len(shapes) == 0:
            # Handle blank images by setting the points to 0
            return 0

        return sum([self._POINTS_FILLED if s.filled else self._POINTS_EMPTY for s in shapes])


class ShapeScaling3x3(SemanticRelationship3x3):
    """
    Generates a semantic relationship that looks at the difference between the scale of shapes from frame to frame.
    """
    # The tolerance for the difference with the expected result, in pixels
    _TOLERANCE = 10

    @property
    def name(self):
        return 'Shape Scaling'

    def generate(self, ravens_matrix, axis):
        # Extract the shapes for the first and second image
        shapes_1 = _extractor.apply(self._select_first_image(ravens_matrix, axis))
        shapes_2 = _extractor.apply(self._select_second_image(ravens_matrix, axis))

        # Now compute the total size of all the shapes
        size_1 = self._compute_total_size(shapes_1)
        size_2 = self._compute_total_size(shapes_2)

        # The expected result is the difference between the total sizes
        return size_1 - size_2

    def test(self, expected, ravens_matrix, answers, axis):
        # Get the shapes for the second image and compute their total size
        shapes = _extractor.apply(self._select_second_image(ravens_matrix, axis))
        size = self._compute_total_size(shapes)

        # For each answer, extract the shapes and compute their total size
        for index, answer in enumerate(answers):
            answer_shapes = _extractor.apply(answer)
            answer_size = self._compute_total_size(answer_shapes)

            # If the difference between the image's and the answer's points equals the expected,
            # with a certain degree of error, then we have found an answer
            diff = size - answer_size
            if abs(diff - expected) <= self._TOLERANCE:
                return index + 1

        return None

    def _compute_total_size(self, shapes):
        if len(shapes) == 0:
            # Handle blank images by setting the size to 0
            return 0

        return sum([s.arclength for s in shapes])


class InvertedDiagonalUnion(SemanticRelationship3x3):
    """
    Generates a semantic relationship where the image 'C' is always merged with the image 'G' of a 3x3 matrix.
    This relationship is particularly tailored to solve Basic Problem C-12 as discussed in a Piazza post.
    However, my agent actually discovered that Challenge Problem C-04 can also be solved with this relationship!
    """
    _SIMILARITY_THRESHOLD = 0.9

    @property
    def name(self):
        return 'Inverted Diagonal Union'

    def generate(self, ravens_matrix, axis):
        # Irrespective of the axis, join image 'C' with image 'G'
        image_c = np.array(ravens_matrix[0][2])
        image_g = np.array(ravens_matrix[2][0])

        # The union operation as defined by Kunda in his doctoral dissertation
        # Reference: https://smartech.gatech.edu/bitstream/handle/1853/47639/kunda_maithilee_201305_phd.pdf
        # Here we use minimum instead of maximum because Kunda assumed that the images had a value of 0 for white
        # but, in reality, 0 indicates a black pixel and 255 (or 1 if the image is binary) is white
        merged = np.minimum(image_c, image_g)

        # Reconstruct the image back to its Pillow representation
        return Image.fromarray(merged)

    def test(self, expected, ravens_matrix, answers, axis):
        # For each answer, find the one that matches the expected image most closely based on a defined threshold
        for index, answer in enumerate(answers):
            if _similarity(expected, answer) >= self._SIMILARITY_THRESHOLD:
                return index + 1

        return None


def _similarity(image, other_image):
    # Computes the similarity between this image and another one using the Normalized Root Mean Squared Error
    # References:
    # - https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    # - https://tinyurl.com/ybmwlsur
    image = np.array(_resize(image)).astype(np.int)
    other_image = np.array(_resize(other_image)).astype(np.int)

    rmse = np.sqrt(np.mean((image - other_image) ** 2))
    max_val = max(np.max(image), np.max(other_image))
    min_val = min(np.min(image), np.min(other_image))

    # `max_val - min_val == 0` happens when the two images are either all black or all white
    return 1 - rmse if max_val - min_val == 0 else 1 - (rmse / (max_val - min_val))


def _resize(image):
    # Reduce size of the image to 32 x 32 so that operations are faster, and similarities are easier to find
    return image.resize((32, 32), resample=Image.BICUBIC)
