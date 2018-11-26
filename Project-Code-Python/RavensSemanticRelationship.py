from abc import ABCMeta, abstractproperty, abstractmethod
from collections import defaultdict
from operator import attrgetter

import numpy as np
from PIL import Image, ImageDraw

from RavensShape import (ABOVE, BELOW, INSIDE, LEFT_OF, RIGHT_OF, RavensRelativeSizeRanker, RavensShapeExtractor,
                         RavensShapeIdentifier, RavensShapeMatcher)
from RavensTransformation import InvertedXORTransformation

# Keep singleton instances available to all semantic relationships
_extractor = RavensShapeExtractor()
_matcher = RavensShapeMatcher()
_identifier = RavensShapeIdentifier()

# The center of an image inside a Raven's matrix
_cx, _cy = 184 / 2, 184 / 2


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
            match = _best_shape_match(b, shapes_a)

            if not match:
                continue

            b.label = match.label

        for c in shapes_c:
            match = _best_shape_match(c, shapes_a)

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
                match = _best_shape_match(expected_shape, answer_shapes)

                if not match:
                    # This answer does not comply with all the expected shapes
                    all_match = False
                    break

            if all_match:
                # We have found an answer that complies with all the expected shapes!
                return index + 1

        return None


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

        # We constrain this relationship to work only for problems where all shapes are the same
        # to avoid answering other problems incorrectly since this relationship was very particular
        # of the problems in the Basic Set C only
        if not self._all_shapes_are_equal(shapes_1 + shapes_2):
            # We return a garbage number that will not match with anything
            return -9999

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

    def _all_shapes_are_equal(self, shapes):
        return len(set([s.shape for s in shapes])) == 1

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
        if not self._validate(ravens_matrix):
            return None

        # Irrespective of the axis, join image 'C' with image 'G'
        image_c = np.array(ravens_matrix[0][2])
        image_g = np.array(ravens_matrix[2][0])

        merged = self._union(image_c, image_g)

        # Reconstruct the image back to its Pillow representation
        return Image.fromarray(merged)

    def test(self, expected, ravens_matrix, answers, axis):
        if expected is None:
            # The relationship does not apply to this problem so no answer can be given
            return None

        # For each answer, find the one that matches the expected image most closely based on a defined threshold
        for index, answer in enumerate(answers):
            if _similarity(expected, answer) >= self._SIMILARITY_THRESHOLD:
                return index + 1

        return None

    def is_valid(self, axis):
        # Only valid for rows, since the same behavior applies for columns and diagonals
        return axis == 0

    def _validate(self, matrix):
        # Validate this relationship applies to this problem by confirming that 'B' + 'D' = 'E'
        image_b = np.array(matrix[0][1])
        image_d = np.array(matrix[1][0])
        image_e = matrix[1][1]

        merged = Image.fromarray(self._union(image_b, image_d))

        return _similarity(merged, image_e) >= self._SIMILARITY_THRESHOLD

    def _union(self, image_1, image_2):
        # The union operation as defined by Kunda in his doctoral dissertation
        # Reference: https://smartech.gatech.edu/bitstream/handle/1853/47639/kunda_maithilee_201305_phd.pdf
        # Here we use minimum instead of maximum because Kunda assumed that the images had a value of 0 for white
        # but, in reality, 0 indicates a black pixel and 255 (or 1 if the image is binary) is white
        return np.minimum(image_1, image_2)


class FindMissingFrame(SemanticRelationship3x3):
    """
    Generates a semantic relationship that finds the missing frame in the row or column.
    """
    _SIMILARITY_THRESHOLD = 0.85

    @property
    def name(self):
        return 'Find Missing Frame'

    def generate(self, ravens_matrix, axis):
        # The row or column of the expected frames
        if axis == 0:
            expected_frames = [ravens_matrix[0][0], ravens_matrix[0][1], ravens_matrix[0][2]]
        else:
            expected_frames = [ravens_matrix[0][0], ravens_matrix[1][0], ravens_matrix[2][0]]

        expected_frames_indices = {0, 1, 2}

        image_1 = self._select_first_image(ravens_matrix, axis)
        image_2 = self._select_second_image(ravens_matrix, axis)

        found_frames_indices = set()

        # Find the match for the first image which would 'G' for row or 'C' for column
        found_frames_indices.add(self._find_match(image_1, expected_frames))
        # Find the match for the second image which would 'H' for row or 'F' for column
        found_frames_indices.add(self._find_match(image_2, expected_frames))

        # Compute the number of missing indices that were not matched
        missing_frame_indices = expected_frames_indices - found_frames_indices

        if len(missing_frame_indices) > 1:
            # If this happens, it means the relationship is most likely not applicable
            # we return a black image which most likely will not match with anything
            return Image.new(image_1.mode, image_1.size)

        # The generated solution is the missing frame that must match an answer
        return expected_frames[list(missing_frame_indices)[0]]

    def test(self, expected, ravens_matrix, answers, axis):
        # For each answer, find the one that matches the expected image most closely based on a defined threshold
        similarities = [_similarity(expected, answer) for answer in answers]
        best = np.argmax(similarities)

        return best + 1 if similarities[best] >= self._SIMILARITY_THRESHOLD else None

    def is_valid(self, axis):
        # Not valid for diagonals
        return axis != 2

    def _find_match(self, image, expected_frames):
        # Find the expected frame that matches the given image
        for index, expected_frame in enumerate(expected_frames):
            if _similarity(image, expected_frame) >= self._SIMILARITY_THRESHOLD:
                return index

        return -1


class FindAndMergeCommonShapesRowColumn(SemanticRelationship3x3):
    """
    Generates a semantic relationships that finds the commons shapes row-wise and column-wise, and merges them together.
    """
    _SIMILARITY_THRESHOLD = 0.85

    @property
    def name(self):
        return 'Find And Merge Common Shapes Row Column'

    def generate(self, ravens_matrix, axis):
        # Extract the shapes for frames 'G' and 'H'
        shapes_g = _extractor.apply(ravens_matrix[2][0])
        shapes_h = _extractor.apply(ravens_matrix[2][1])

        # Extract the shapes for frames 'C' and 'F'
        shapes_c = _extractor.apply(ravens_matrix[0][2])
        shapes_f = _extractor.apply(ravens_matrix[1][2])

        # Find the common shapes between 'G' and 'H', and 'C' and 'F'
        common_shapes_g_h = self._find_common_shapes(shapes_g, shapes_h)
        common_shapes_c_f = self._find_common_shapes(shapes_c, shapes_f)

        if len(common_shapes_g_h) == 0 or len(common_shapes_c_f) == 0:
            # If there are not common shapes in the row or in the column,
            # then this relationship must likely does not apply to the problem
            return None

        # The expected answer is a reconstructed image of the merge of the common shapes of the row and column
        reconstructed = self._reconstruct(common_shapes_g_h + common_shapes_c_f)

        return reconstructed

    def test(self, expected, ravens_matrix, answers, axis):
        if expected is None:
            # This relationship does not apply and no answer can be given
            return None

        # For each answer, find the one that matches the expected image most closely based on a defined threshold
        similarities = [_similarity(expected, answer) for answer in answers]
        best = np.argmax(similarities)

        return best + 1 if similarities[best] >= self._SIMILARITY_THRESHOLD else None

    def is_valid(self, axis):
        # Only valid for rows, since the same behavior applies for columns and diagonals
        return axis == 0

    def _find_common_shapes(self, shapes, other_shapes):
        # Finds the common shapes between the given list of shapes and the list of the other shapes
        return [
            shape
            for shape in shapes
            if self._has_match(shape, other_shapes)
        ]

    def _has_match(self, shape, other_shapes):
        # Determines whether the given shape has a match out of the list of the other shapes using a full matching
        similarities = [_matcher.apply(shape, other_shape, match=_matcher.MATCH_ALL) for other_shape in other_shapes]
        match = np.argmax(similarities)

        return similarities[match] >= self._SIMILARITY_THRESHOLD

    def _reconstruct(self, shapes):
        reconstructed = Image.new('L', (184, 184), 255)
        draw = ImageDraw.Draw(reconstructed)

        # This most likely means this relationship does not apply to this problem since we are expecting common shapes
        if len(shapes) == 0:
            return None

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
            # Add an extra line over the contour to darken it so that it helps with matching
            draw.line(shape.contour.flatten().tolist(), fill=0, width=3)

        return reconstructed


class FindMissingCenterShapeAndApplyPattern(SemanticRelationship3x3):
    """
    Generates a semantic relationships that looks for the missing center shape in the complete Raven's matrix and then
    applies a specific pattern to it.

    This relationship solves specifically Basic Problem D-06.
    """
    _SIMILARITY_THRESHOLD = 0.9
    _KEEP_ALL = 'KeepALL'
    _REMOVE_LARGEST = 'RemoveLargest'
    _PATTERNS = [_KEEP_ALL, _REMOVE_LARGEST]

    @property
    def name(self):
        return 'Find Missing Center Shape and Apply Pattern'

    def generate(self, ravens_matrix, axis):
        shapes_per_frame = _convert_matrix_to_shapes(ravens_matrix)
        center_shapes = _find_center_shapes(shapes_per_frame)

        if len(center_shapes) == 0:
            # No center shapes means this relationship does not apply to the given Raven's matrix
            # so we return `None` to make sure the tester knows it is invalid
            return None

        valid_pattern = self._find_valid_pattern(shapes_per_frame, center_shapes, axis)

        if valid_pattern is None:
            # This relationship does not apply to the given Raven's matrix
            return None

        # Once the valid pattern has been found, we can find the missing center shape
        missing_shape = _find_missing_center_shape(center_shapes)

        if missing_shape is None:
            # The patterns applies, but the relationship is actually not valid!
            return None

        # Now that we have the missing shape, we can apply the pattern to the last image
        # and join it with the missing shape to produce the set of expected shapes
        # Either 'H' for row-wise or 'F' for column-wise
        shapes = shapes_per_frame[7] if axis == 0 else shapes_per_frame[5]
        center_shape = center_shapes[7] if axis == 0 else center_shapes[5]

        return self._apply_pattern(shapes, center_shape, missing_shape, valid_pattern)

    def test(self, expected, ravens_matrix, answers, axis):
        if expected is None:
            # The relationship was invalid, so no answer can be provided
            return None

        # For each answer find the one that complies with all the expected shapes
        for index, answer in enumerate(answers):
            answer_shapes = _extractor.apply(answer)
            if self._shapes_match(expected, answer_shapes):
                # We have found the answer
                return index + 1

        return None

    def is_valid(self, axis):
        # Not valid for diagonals
        return axis != 2

    def _find_valid_pattern(self, shapes_per_frame, center_shapes, axis):
        # Validate each pattern and find the one that applies based on the axis
        valid_pattern = None

        for pattern in self._PATTERNS:
            if self._is_valid_pattern(shapes_per_frame, center_shapes, axis, pattern):
                # We take the first valid pattern
                valid_pattern = pattern
                break

        return valid_pattern

    def _is_valid_pattern(self, shapes_per_frame, center_shapes, axis, pattern):
        # Either 'G' for row-wise or 'C' for column-wise
        shapes_first = shapes_per_frame[6] if axis == 0 else shapes_per_frame[2]
        center_shape_first = center_shapes[6] if axis == 0 else center_shapes[2]

        # Either 'H' for row-wise or 'F' for column-wise
        shapes_second = shapes_per_frame[7] if axis == 0 else shapes_per_frame[5]
        center_shape_second = center_shapes[7] if axis == 0 else center_shapes[5]

        # Apply the pattern to generate a set of expected shapes for the first image
        # using the center shape of the second image as the "missing" one
        expected_shapes_first = self._apply_pattern(shapes_first, center_shape_first, center_shape_second, pattern)

        # Test the pattern against the second image
        return self._shapes_match(expected_shapes_first, shapes_second)

    def _apply_pattern(self, shapes, center_shape, missing_shape, pattern):
        if pattern == self._KEEP_ALL:
            # Keep all shapes except the center one
            shapes_to_keep = [
                shape
                for shape in shapes
                if shape.label != center_shape.label
            ]
        elif pattern == self._REMOVE_LARGEST:
            # Remove the largest of the shapes and keep all the other one, except the center one again
            largest_rank = max([s.size_rank for s in shapes])
            shapes_to_keep = [
                shape
                for shape in shapes
                if shape.label != center_shape.label and shape.size_rank != largest_rank
            ]
        else:
            raise ValueError('Invalid pattern: {}'.format(pattern))

        # The expected shapes are the shapes after the pattern was applied and the missing center shape
        return shapes_to_keep + [missing_shape]

    def _shapes_match(self, shapes, other_shapes):
        # Compares two sets of shapes and determines whether they all match or not
        # Verify they both have the same number of shapes
        if len(shapes) != len(other_shapes):
            return False

        all_match = True

        # For each of the shapes make sure the other shapes contains it too
        for shape in shapes:
            match = _best_shape_match(shape, other_shapes)

            if not match:
                # This answer does not comply with all the expected shapes
                all_match = False
                break

        return all_match


class FindMissingCenterShapeAndMissingPattern(SemanticRelationship3x3):
    """
    Generates a semantic relationships that looks for the missing center shape in the complete Raven's matrix and also
    finds the missing pattern, out of some known given set.

    This relationship is for problems Basic Problem D-07 and Basic Problem D-08.
    """
    NO_PATTERN = -1
    # Patterns for Basic Problem D-07
    SURROUNDED_BY_SHAPES = 0
    INSIDE_OTHER_SHAPE = 1
    INSIDE_SAME_SHAPE = 2
    # Patterns for Basic Problem D-08
    FILLED_SHAPE = 3
    EMPTY_SHAPE = 4

    def __init__(self, pattern_set):
        assert len(pattern_set) == 3, 'A set of 3 patterns must be provided!'
        self._pattern_set = set(pattern_set)

    @property
    def name(self):
        return 'Find Missing Center Shape and Missing Pattern'

    def generate(self, ravens_matrix, axis):
        shapes_per_frame = _convert_matrix_to_shapes(ravens_matrix)
        center_shapes = _find_center_shapes(shapes_per_frame)

        if len(center_shapes) == 0:
            # This relationship does not apply for the given Raven's matrix
            return None

        pattern_set_is_valid = self._validate_pattern_set(shapes_per_frame, center_shapes)

        if not pattern_set_is_valid:
            # This relationship must likely does not match this problem
            return None

        # Now that we have validated the pattern set can be applied to this problem,
        # we can find the missing center shape and the missing pattern in the last row
        missing_shape = _find_missing_center_shape(center_shapes)

        if missing_shape is None:
            # The pattern set applies, but the relationship is actually not valid!
            return None

        missing_pattern = self._find_missing_pattern(shapes_per_frame, center_shapes)

        if missing_pattern == self.NO_PATTERN:
            # The pattern set applies, but no pattern was found, should not happen!
            return None

        # The expected result is a tuple of the missing shape and the missing pattern
        return missing_shape, missing_pattern

    def test(self, expected, ravens_matrix, answers, axis):
        if expected is None:
            # The relationship was not valid so no answer can be given
            return None

        missing_shape, missing_pattern = expected

        # For each answer, find the one that complies with the missing center shape and the missing pattern
        for index, answer in enumerate(answers):
            answer_shapes = _extractor.apply(answer)
            answer_center_shape = _find_center_shape(answer_shapes)

            # First, validate the missing center shape is the same
            if missing_shape.shape != answer_center_shape.shape:
                continue

            # Then, validate the answer has the missing pattern
            answer_pattern = self._find_pattern(answer_center_shape, answer_shapes)
            if answer_pattern != missing_pattern:
                continue

            # We have found an answer than complies with both the missing center shape and the missing pattern!
            return index + 1

        return None

    def is_valid(self, axis):
        # Only valid for rows since it's the same behavior for columns and diagonals
        return axis == 0

    def _validate_pattern_set(self, shapes_per_frame, center_shapes):
        # Validate the provided pattern set by evaluating it against the first row and the first column

        # Row patterns
        patterns = [self._find_pattern(center_shapes[i], shapes_per_frame[i]) for i in [0, 1, 2]]
        # The patterns found should match the provided set
        row_is_valid = len(patterns) == len(self._pattern_set) and len(self._pattern_set - set(patterns)) == 0

        # Column patterns
        patterns = [self._find_pattern(center_shapes[i], shapes_per_frame[i]) for i in [0, 3, 6]]
        column_is_valid = len(patterns) == len(self._pattern_set) and len(self._pattern_set - set(patterns)) == 0

        return row_is_valid and column_is_valid

    def _find_missing_pattern(self, shapes_per_frame, center_shapes):
        # Finds the missing pattern, out of the provided set, in the last row of the Raven's matrix
        shapes_g = shapes_per_frame[6]
        center_shape_g = center_shapes[6]

        shapes_h = shapes_per_frame[7]
        center_shape_h = center_shapes[7]

        # Find the pattern in 'G'
        pattern_g = self._find_pattern(center_shape_g, shapes_g)
        # Find the pattern in 'H'
        pattern_h = self._find_pattern(center_shape_h, shapes_h)

        # Sanity check: the patterns must not be the same!
        if pattern_g == pattern_h:
            return self.NO_PATTERN

        # The missing pattern is the difference between the set of patterns found and the provided set
        missing_pattern = list(self._pattern_set - {pattern_g, pattern_h})

        # Another sanity check: there can't be more than 1 missing pattern!
        if len(missing_pattern) > 1:
            return self.NO_PATTERN

        return missing_pattern[0]

    def _find_pattern(self, center_shape, all_shapes):
        # Finds the pattern that the given shapes describe, out of the known ones
        if center_shape is None:
            return self.NO_PATTERN

        if self._is_surrounded_by_shapes(center_shape):
            return self.SURROUNDED_BY_SHAPES

        if self._is_inside_other_shape(center_shape, all_shapes):
            return self.INSIDE_OTHER_SHAPE

        if self._is_inside_same_shape(center_shape, all_shapes):
            return self.INSIDE_SAME_SHAPE

        if self._is_filled(center_shape):
            return self.FILLED_SHAPE

        if self._is_empty(center_shape):
            return self.EMPTY_SHAPE

        return self.NO_PATTERN

    def _is_surrounded_by_shapes(self, shape):
        # Determines whether the given shape is surrounded by other shapes
        positions = shape.positions

        # A shape is surrounded by others under the following conditions:
        # - It has a shape above and below (surrounded by 2 vertically)
        # - It has a shape right and left of it (surrounded by 2 horizontally)
        # - It has a shape in all four positions (surrounded by 4)
        return (BELOW in positions and ABOVE in positions or
                LEFT_OF in positions and RIGHT_OF in positions)

    def _is_inside_other_shape(self, shape, all_shapes):
        # Determines whether the given shape is inside another **different** shape
        parent_shapes = self._find_parent_shapes(shape, all_shapes)

        if parent_shapes is None:
            return False

        # Make sure all the parent shapes are **different** to the given shape
        for parent_shape in parent_shapes:
            if parent_shape.shape == shape.shape:
                return False

        return True

    def _is_inside_same_shape(self, shape, all_shapes):
        # Determines whether the given shape is inside another **same** shape
        parent_shapes = self._find_parent_shapes(shape, all_shapes)

        if parent_shapes is None:
            return False

        # Make sure all the parent shapes are the **same** as the given shape
        for parent_shape in parent_shapes:
            if parent_shape.shape != shape.shape:
                return False

        return True

    def _is_filled(self, shape):
        # Determines whether the given shape is filled
        return shape.filled

    def _is_empty(self, shape):
        # Determines whether the given shape is empty, i.e. not filled
        return not shape.filled

    def _find_parent_shapes(self, shape, all_shapes):
        # Finds the parent shapes of the given shape
        if INSIDE not in shape.positions:
            return None

        # Find the shapes inside of which this shape is
        return list(filter(lambda x: x.label in shape.positions[INSIDE], all_shapes))


class FindMissingImagePattern(SemanticRelationship3x3):
    """
    Generates a semantic relationship that looks for the missing image pattern that should be valid in groups of three
    for each image in the Raven's matrix. In other words, image 'A' should have two other pairs that match the result of
    applying the patterns, at the image level.

    For problem Basic Problem D-09.
    """
    _SIMILARITY_THRESHOLD = 0.9

    # Pattern for Basic Problem D-09
    ROTATION_AND_UNION = 0

    def __init__(self, pattern):
        self._pattern = pattern

    @property
    def name(self):
        return 'Find Missing Image Pattern'

    def generate(self, ravens_matrix, axis):
        for row in ravens_matrix:
            for frame in row:
                # Apply the pattern to each frame in the Raven's matrix
                unmatched = self._apply_pattern(frame, ravens_matrix)

                if unmatched is None:
                    # This relationship must likely does not apply to this problem
                    return None

                if len(unmatched) == 1:
                    # We have found the missing pattern!
                    return unmatched[0]

        # No missing patterns, which means this relationship probably does not apply
        return None

    def test(self, expected, ravens_matrix, answers, axis):
        if expected is None:
            # This relationship is not valid so no answer can be given
            return None

        # For each answer find the one that matches the expected missing one
        similarities = [_similarity(expected, answer) for answer in answers]
        best = np.argmax(similarities)

        return best + 1 if similarities[best] >= self._SIMILARITY_THRESHOLD else None

    def is_valid(self, axis):
        # Only valid for rows since it's the same behavior for columns and diagonals
        return axis == 0

    def _apply_pattern(self, image, matrix):
        # Applies the pattern to the given image (or frame) of the Raven's matrix
        unmatched = []
        if self._pattern == self.ROTATION_AND_UNION:
            # Rotate and merge the image
            rotated = self._apply_rotation(image)
            merged = self._apply_union(image, rotated)

            # For this pattern, the image must have a match for its rotated version
            # and a match for the union of the image with the rotated version
            if not self._has_match(rotated, matrix):
                unmatched.append(rotated)

            if not self._has_match(merged, matrix):
                unmatched.append(merged)
        else:
            raise ValueError('Invalid pattern: {}!'.format(self._pattern))

        # Return `None` for those case where there are more than one pattern missing
        # since that means this problem does not fit with this relationship
        return unmatched if len(unmatched) <= 1 else None

    def _apply_rotation(self, image):
        # Rotates the image
        return Image.fromarray(np.rot90(image))

    def _apply_union(self, image_1, image_2):
        # The union operation as defined by Kunda in his doctoral dissertation
        # Reference: https://smartech.gatech.edu/bitstream/handle/1853/47639/kunda_maithilee_201305_phd.pdf
        # Here we use minimum instead of maximum because Kunda assumed that the images had a value of 0 for white
        # but, in reality, 0 indicates a black pixel and 255 (or 1 if the image is binary) is white
        return Image.fromarray(np.minimum(image_1, image_2))

    def _has_match(self, image, matrix):
        # Checks to see if the given image can be found in the other frames of the Raven's matrix
        for row in matrix:
            for frame in row:
                if self._images_match(image, frame):
                    return True

        return False

    def _images_match(self, image_1, image_2):
        return _similarity(image_1, image_2) >= self._SIMILARITY_THRESHOLD


class FindMissingShapeAndCount(SemanticRelationship3x3):
    """
    Generates a semantic relationship that finds the missing shape and its count in the Raven's matrix.
    This relationship is specifically for problem Basic Problem D-12.
    """

    @property
    def name(self):
        return 'Find Missing Shape and Count'

    def generate(self, ravens_matrix, axis):
        shape_counts = defaultdict(int)

        # For each frame in the Raven's matrix, extract its shape and count
        for row in ravens_matrix:
            for frame in row:
                result = self._count(frame)

                if result is None:
                    # This relationship do not apply to this problem
                    return None

                # Accumulate the count by shape
                shape, count = result
                shape_counts[shape] += count

        # Count the number of occurrences for each count
        common_counts = defaultdict(int)
        for count in shape_counts.values():
            common_counts[count] += 1

        # If there are more than 2 common counts, then this problem does not
        # fit with this relationship because only one count should be different
        if len(common_counts.keys()) != 2:
            return None

        # Find the count that has less occurrences, this is the count that has
        # some of its elements missing
        count_with_least_occurrences = min(common_counts, key=common_counts.get)
        count_with_most_occurrences = max(common_counts, key=common_counts.get)

        # Now compute the number that is missing which should be the difference
        # between the count with the least occurrences and the count with the most
        # occurrences
        missing_count = abs(count_with_most_occurrences - count_with_least_occurrences)

        # Finally, find the shape that has the count with the least occurrences,
        # this is the shape that needs to be present in the answer
        missing_shape = None
        for shape, count in shape_counts.iteritems():
            if count == count_with_least_occurrences:
                missing_shape = shape

        # Sanity check: if no missing shape was found, this relationship most likely
        # does not apply to this problem
        if missing_shape is None:
            return None

        # The expected result is the tuple of missing shape and missing count
        return missing_shape, missing_count

    def test(self, expected, ravens_matrix, answers, axis):
        if expected is None:
            # The relationship was not valid so no answer can be given
            return None

        missing_shape, missing_count = expected

        # For each answer, find the one that complies with the missing shape and the missing count
        for index, answer in enumerate(answers):
            answer_result = self._count(answer)

            if answer_result is None:
                continue

            answer_common_shape, answer_count = answer_result

            # First, validate the missing shape is the same
            if missing_shape != answer_common_shape:
                continue

            # Then, validate the answer has the missing count
            if answer_count != missing_count:
                continue

            # We have found an answer than complies with both the missing shape and the missing count!
            return index + 1

        return None

    def is_valid(self, axis):
        # Only valid for rows since it's the same behavior for columns and diagonals
        return axis == 0

    def _count(self, frame):
        # Extracts the common shape of the frame and its count
        shapes = _extractor.apply(frame)

        if len(shapes) == 0:
            # No shapes were found which means this problem is not adequate for this relationship
            return None

        # Validate that all shapes inside a frame are the same
        if not self._all_shapes_are_equal(shapes):
            # If they are not, this relationship most likely does not apply with this pattern
            return None

        # Use the RavensShapeIdentifier, which is based on common components, to count the shapes
        # the reason being that the RavensShapeExtractor sometimes extracts less shapes because
        # they are too close to each other; however, that works for most problems so changing the
        # logic in the extractor will probably affect a lot of other problems (overfitting much?)
        # so it is a safer bet to count using the identifier's algorithm.
        identified_shapes = _identifier.apply(frame)

        return self._extract_common_shape(shapes), len(identified_shapes)

    def _all_shapes_are_equal(self, shapes):
        # Checks that all shapes, in the given list, are all equal
        return len(set([s.shape for s in shapes])) == 1

    def _extract_common_shape(self, shapes):
        # Extracts the common shape of a list of shapes, which is assumed to be all equal shapes
        return list(set([s.shape for s in shapes]))[0]


class DeleteCommonShapesAndKeepCenterShape(SemanticRelationship3x3):
    """
    Generates a relationship that deletes the common shape between two images, but keeps the center shape.
    For problem Basic Problem E-06.
    """
    _SIMILARITY_THRESHOLD = 0.9

    def __init__(self):
        self._inverted_xor_transformation = InvertedXORTransformation()

    @property
    def name(self):
        return 'Delete Common Shapes and Keep Center Shape'

    def generate(self, ravens_matrix, axis):
        # First, find the center shape in image 'G'
        shapes_g = _extractor.apply(ravens_matrix[2][0])
        center_shape = _find_center_shape(shapes_g)

        if center_shape is None:
            # No center shape means this relationship most likely does not apply
            return None

        # Then, delete the common shapes by applying an inverter XOR transformation
        # between images 'G' and 'H'
        xor_g_h = self._inverted_xor_transformation.apply(ravens_matrix[2][0], other=ravens_matrix[2][1])

        # Finally, reconstruct the image by adding the center shape back
        reconstructed = self._reconstruct(xor_g_h, center_shape)

        return reconstructed

    def test(self, expected, ravens_matrix, answers, axis):
        if expected is None:
            # Invalid relationship so no answer can be given
            return None

        # For each answer, find the one that matches the expected image most closely based on a defined threshold
        similarities = [_similarity(expected, answer) for answer in answers]
        best = np.argmax(similarities)

        return best + 1 if similarities[best] >= self._SIMILARITY_THRESHOLD else None

    def is_valid(self, axis):
        # Only valid for rows since it's the same behavior for columns and diagonals
        return axis == 0

    def _reconstruct(self, image, center_shape):
        reconstructed = image.copy()
        draw = ImageDraw.Draw(reconstructed)

        # Reconstruct the image by drawing the center shape
        draw.polygon(center_shape.contour.flatten().tolist(), fill=0 if center_shape.filled else 255, outline=0)
        # Add an extra line over the contour to darken it so that it helps with matching
        draw.line(center_shape.contour.flatten().tolist(), fill=0, width=3)

        return reconstructed


class ShapeCountAndAnglePointsSystem(SemanticRelationship3x3):
    """
    Generates a semantic relationship that counts the difference in number of shapes between two images and also applies
    a points system based on the angles of the shapes inside those images.

    For Problem Basic E-12.
    """
    _POINTS_ANGLE_1 = 1
    _POINTS_ANGLE_2 = -1
    _POINTS_INVALID_ANGLE = -9999

    @property
    def name(self):
        return 'Shape Count and Angle Points System'

    def generate(self, ravens_matrix, axis):
        # Extract the shapes for the first and second image
        shapes_1 = _extractor.apply(self._select_first_image(ravens_matrix, axis))
        shapes_2 = _extractor.apply(self._select_second_image(ravens_matrix, axis))

        # We constrain this relationship to work only for problems where all shapes are the same
        # to avoid answering other problems incorrectly
        if not self._all_shapes_are_equal(shapes_1 + shapes_2):
            return None

        # Extract the rounded angles for both images and join them
        angles = list(set(self._extract_rounded_angles(shapes_1) + self._extract_rounded_angles(shapes_2)))

        if len(angles) == 0 or len(angles) > 2:
            # If there are no angles or there are more than 2 distinct angles, then this
            # relationship most likely does not apply to this problem
            return None

        # Now count the points for each image based on the system
        points_1 = self._count_points(shapes_1, angles)
        points_2 = self._count_points(shapes_2, angles)

        # The expected result is the difference in count of shapes, the sum of points and the angles used
        return len(shapes_1) - len(shapes_2), points_1 + points_2, angles

    def test(self, expected, ravens_matrix, answers, axis):
        if expected is None:
            # Invalid relationship so no answer can be given
            return None

        expected_count_shapes, expected_points, angles = expected

        # For each answer, extract the shapes and rounded angles, and count their points
        for index, answer in enumerate(answers):
            answer_shapes = _extractor.apply(answer)

            # First, check the count of shapes matches
            if len(answer_shapes) != expected_count_shapes:
                continue

            answer_angles = self._extract_rounded_angles(answer_shapes)

            # Then, validate that the angles of the answer are in the set of angles used
            # to generate the expected result in order to ensure consistency
            if not set(answer_angles).issubset(set(angles)):
                continue

            # Finally, check the points of the given answer using those same angles
            answer_points = self._count_points(answer_shapes, angles)

            if answer_points == expected_points:
                # We have found the answer whose points match the expected result!
                return index + 1

        return None

    def is_valid(self, axis):
        # Only valid for rows and columns
        return axis != 2

    def _all_shapes_are_equal(self, shapes):
        return len(set([s.shape for s in shapes])) == 1

    def _extract_rounded_angles(self, shapes):
        # Extracts and returns the rounded angles to the nearest multiple of 10
        return [
            self._round_to_nearest_multiple_of_ten(s.angle)
            for s in shapes
        ]

    def _count_points(self, shapes, angles):
        if len(shapes) == 0:
            # Handle blank images by setting the points to 0
            return 0

        points = 0

        # Compute the points based on the rounded angle of each shape
        for s in shapes:
            rounded_angle = self._round_to_nearest_multiple_of_ten(s.angle)

            if rounded_angle == angles[0]:
                points += self._POINTS_ANGLE_1
            elif rounded_angle == angles[1]:
                points += self._POINTS_ANGLE_2
            else:
                points += self._POINTS_INVALID_ANGLE

        return points

    def _round_to_nearest_multiple_of_ten(self, number):
        # Rounds a number to the nearest multiple of 10
        # Smaller multiple
        smaller_multiple = (number // 10) * 10
        # Larger multiple
        larger_multiple = smaller_multiple + 10

        # Return the closes of the two
        return larger_multiple if number - smaller_multiple > larger_multiple - number else smaller_multiple


def _convert_matrix_to_shapes(matrix):
    # Extracts all the shapes from all frames in the matrix
    return [_extractor.apply(frame) for row in matrix for frame in row]


def _find_center_shapes(shapes_per_frame):
    # Finds all center shapes of all frames in the matrix
    center_shapes = []

    for shapes in shapes_per_frame:
        center_shape = _find_center_shape(shapes)

        if center_shape is None:
            # If we can't find one of the center shapes, then return an empty list
            # because it means that this matrix is probably not about center shapes
            return []

        center_shapes.append(center_shape)

    return center_shapes


def _find_center_shape(shapes):
    # Finds the center shape out of a list of shapes
    # Filter the shapes whose centroid is closest to the center of the image
    center_shapes = filter(lambda x: np.sqrt((_cx - x.centroid[0]) ** 2 + (_cy - x.centroid[1]) ** 2) <= 10.0, shapes)

    if len(center_shapes) == 0:
        return None

    # Find the smallest one by sorting the centered shapes in increasing order of `size_rank`
    # This handles cases where there are multiple shapes one inside the other, all technically centered
    # but the true center one is the smallest one inside all the other ones
    center_shapes = sorted(center_shapes, key=attrgetter('size_rank'))
    center_shape = center_shapes[0]

    return center_shape


def _find_missing_center_shape(center_shapes):
    # Finds the missing center shape in the complete Raven's matrix represented by the set of all center shapes
    # Keep a dictionary of the actual shape objects so that the missing one can be retrieved
    center_shapes_instances = {}
    for s in center_shapes:
        if s.shape not in center_shapes_instances:
            center_shapes_instances[s.shape] = s

    # Count each shape occurrence
    shape_counts = defaultdict(int)
    for s in center_shapes:
        shape_counts[s.shape] += 1

    # The missing shape is the one that has two occurrences only
    missing_shape = None
    missing_shapes_count = 0

    for shape, count in shape_counts.items():
        if count == 2:
            missing_shape = shape
            missing_shapes_count += 1

    # If there are more than one missing shape or no missing shape was found,
    # the relationship most likely does not apply
    if missing_shapes_count > 1 or missing_shape is None:
        return None

    # Return the actual missing shape instance
    return center_shapes_instances[missing_shape]


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


def _best_shape_match(shape, other_shapes, similarity_threshold=0.9):
    # Finds the shape that best matches the given shape out of the list of the other shapes using a full matching
    similarities = [_matcher.apply(shape, other_shape, match=_matcher.MATCH_ALL) for other_shape in other_shapes]
    match = np.argmax(similarities)

    return other_shapes[match] if similarities[match] >= similarity_threshold else None


def _shapes_match(shape, other_shape, similarity_threshold=0.9):
    return _matcher.apply(shape, other_shape, match=_matcher.MATCH_ALL) >= similarity_threshold
