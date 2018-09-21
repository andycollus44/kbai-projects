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
    def test(self, expected, answers):
        """
        Tests the generated expected result against the available answers to the problem.

        :param expected: The expected result generated from a call to `generate()`.
        :param answers: The set of available answers.
        :type answers: list[PIL.Image.Image]
        :return: The correct answer or None, if no answer matches the expected results.
        :rtype: int
        """
        pass

    def __repr__(self):
        return self.name


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

    def test(self, expected, answers):
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
