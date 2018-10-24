from abc import ABCMeta, abstractmethod, abstractproperty
from collections import namedtuple
from operator import itemgetter

import numpy as np
from PIL import Image

from RavensTransformation import (SINGLE, MULTI, FlipTransformation, MirrorTransformation, NoOpTransformation,
                                  RotationTransformation, ShapeFillTransformation, UnionTransformation,
                                  XORTransformation)

Answer = namedtuple('Answer', ['similarity', 'answer'])


class RavensProblemSolverFactory:
    def __init__(self):
        pass

    def create(self, problem_type):
        """
        Creates an instance of a RavensProblemSolver for the given problem type.

        :param problem_type: The problem type, either '2x2' or '3x3'.
        :type problem_type: str
        :return: A RavensProblemSolver.
        :rtype: RavensProblemSolver
        """
        if problem_type == '2x2':
            return _Ravens2x2Solver()
        elif problem_type == '3x3':
            return _Ravens3x3Solver()
        else:
            raise ValueError('Invalid problem type: {}'.format(problem_type))


class RavensProblemSolver:
    __metaclass__ = ABCMeta

    _SIMILARITY_THRESHOLD = 0.90

    def run(self, problem):
        """
        Runs this solver to find an answer to the given problem.

        :param problem: The RPM problem to solve.
        :type problem: RavensVisualProblem.RavensVisualProblem
        :return: The index of the selected answer.
        :rtype: int
        """
        # Apply each transformation for each valid axis defined by the underlying solver
        answers = [self._apply(problem, transformation, axis)
                   for axis in self._axes
                   for transformation in self._transforms]
        # Sort them in increasing order by their similarity measure
        answers.sort(key=itemgetter(0), reverse=True)

        # The best answer is the first sorted element
        _, best = answers[0] if answers else (0.0, None)

        return best

    def _are_similar(self, image, other_image):
        return self._similarity(image, other_image) >= self._SIMILARITY_THRESHOLD

    def _similarity(self, image, other_image):
        # Computes the similarity between this image and another one using the Normalized Root Mean Squared Error
        # References:
        # - https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
        # - https://tinyurl.com/ybmwlsur
        image = np.array(self._resize(image)).astype(np.int)
        other_image = np.array(self._resize(other_image)).astype(np.int)

        rmse = np.sqrt(np.mean((image - other_image) ** 2))
        max_val = max(np.max(image), np.max(other_image))
        min_val = min(np.min(image), np.min(other_image))

        # `max_val - min_val == 0` happens when the two images are either all black or all white
        return 1 - rmse if max_val - min_val == 0 else 1 - (rmse / (max_val - min_val))

    def _filter_by_similarity(self, similarities):
        # Filters a list of similarity to keep only the most confident ones
        return filter(lambda similarity: similarity >= self._SIMILARITY_THRESHOLD, similarities)

    def _resize(self, image):
        # Reduce size of the image to 32 x 32 so that operations are faster, and similarities are easier to find
        return image.resize((32, 32), resample=Image.BICUBIC)

    @abstractproperty
    def _transforms(self):
        # The list of all available transformations for this solver
        pass

    @abstractproperty
    def _axes(self):
        # The list of valid axes for this solver
        pass

    @abstractmethod
    def _apply(self, problem, transformation, axis):
        # Applies a particular transformation to the Raven's matrix of images for the given axis
        # where axis = 0 means row-wise and axis = 1 means column-wise, and returns the best
        # answer along with its similarity measure; if no answer is found, then `None` is returned
        pass


class _Ravens2x2Solver(RavensProblemSolver):
    @property
    def _transforms(self):
        return [
            NoOpTransformation(),
            MirrorTransformation(),
            FlipTransformation(),
            ShapeFillTransformation(),
            XORTransformation(),
            RotationTransformation(90)
        ]

    @property
    def _axes(self):
        return [0, 1]

    def _apply(self, problem, transformation, axis):
        if transformation.type is SINGLE:
            similarity, answer = self._apply_single(problem, transformation, axis)
        elif transformation.type is MULTI:
            similarity, answer = self._apply_multi(problem, transformation, axis)
        else:
            raise ValueError('Invalid transformation of type: {}'.format(transformation.type))

        # Answers are one-indexed based
        return Answer(similarity=similarity, answer=answer + 1 if answer is not None else None)

    def _apply_single(self, problem, transformation, axis):
        matrix = problem.matrix
        answers = problem.answers

        # The 'A' image
        image_1 = transformation.apply(matrix[0][0])
        # Either the 'B' image for row-wise or the 'C' image for column-wise
        image_2 = matrix[0][1] if axis == 0 else matrix[1][0]

        # Compare the image with its pair, if they don't match then their is no answer
        if not super(_Ravens2x2Solver, self)._are_similar(image_1, image_2):
            return 0., None

        # Find the answer that most closely matches the transformed image, i.e. generate and test
        # The 'C' image for row-wise or the 'B' image for column-wise
        last_image = transformation.apply(matrix[1][0]) if axis == 0 else transformation.apply(matrix[0][1])
        similarities = [super(_Ravens2x2Solver, self)._similarity(last_image, answer) for answer in answers]
        candidate = np.argmax(similarities)

        return similarities[candidate], candidate

    def _apply_multi(self, problem, transformation, axis):
        matrix = problem.matrix
        answers = problem.answers

        # The 'A' image
        image_1 = matrix[0][0]
        # Either the 'B' image for row-wise or the 'C' image for column-wise
        image_2 = matrix[0][1] if axis == 0 else matrix[1][0]
        # Apply the multi-transformation to these two images to obtain the expected result
        expected = transformation.apply(image_1, other=image_2)

        # The 'C' image for row-wise or the 'B' image for column-wise
        last_image = matrix[1][0] if axis == 0 else matrix[0][1]
        # For each answer, apply the multi-transformation with the last image to generate potential candidates
        # and test its similarity with the expected result, i.e. generate and test
        similarities = [
            super(_Ravens2x2Solver, self)._similarity(expected, transformation.apply(last_image, other=answer))
            for answer in answers
        ]

        # In contrast with the single-transformations, for multi ones, there is no initial check between the first
        # two pairs to validate they are similar since the two images are needed to obtain a transformation and this
        # result is compared to the application of the same transformation against the last image with the answers
        # So, to avoid answering problems incorrectly, filter out the answers that are not significantly similar
        filtered = super(_Ravens2x2Solver, self)._filter_by_similarity(similarities)

        # If there are no significantly similar candidates, then there is no answer
        if len(filtered) == 0:
            return 0., None

        # Find the best candidate based on similarity
        candidate = np.argmax(similarities)

        return similarities[candidate], candidate


class _Ravens3x3Solver(RavensProblemSolver):
    @property
    def _transforms(self):
        return [
            NoOpTransformation(),
            MirrorTransformation(),
            UnionTransformation()
        ]

    @property
    def _axes(self):
        return [0, 1, 2]

    def _apply(self, problem, transformation, axis):
        if transformation.type is SINGLE:
            similarity, answer = self._apply_single(problem, transformation, axis)
        elif transformation.type is MULTI:
            similarity, answer = self._apply_multi(problem, transformation, axis)
        else:
            raise ValueError('Invalid transformation of type: {}'.format(transformation.type))

        # Answers are one-indexed based
        return Answer(similarity=similarity, answer=answer + 1 if answer is not None else None)

    def _apply_single(self, problem, transformation, axis):
        matrix = problem.matrix
        answers = problem.answers

        if not self._is_single_transformation_valid(matrix, transformation, axis):
            # If the transformation is not applicable, then there is no answer
            return 0., None

        # Apply the given transformation to the first image
        image_1 = transformation.apply(self._select_first_image(matrix, axis))

        # Find the answer that most closely matches the transformed image, i.e. generate and test
        similarities = [super(_Ravens3x3Solver, self)._similarity(image_1, answer) for answer in answers]
        candidate = np.argmax(similarities)

        return similarities[candidate], candidate

    def _apply_multi(self, problem, transformation, axis):
        matrix = problem.matrix
        answers = problem.answers

        if not self._is_multi_transformation_valid(matrix, transformation, axis):
            # If the transformation is not applicable, then there is no answer
            return 0., None

        image_1 = self._select_first_image(matrix, axis)
        image_2 = self._select_second_image(matrix, axis)
        # Apply the multi-transformation to these two images to obtain the expected result
        expected = transformation.apply(image_1, other=image_2)

        # For each answer, test its similarity with the expected result, i.e. generate and test
        similarities = [super(_Ravens3x3Solver, self)._similarity(expected, answer) for answer in answers]

        # Find the best candidate based on similarity
        candidate = np.argmax(similarities)

        return similarities[candidate], candidate

    def _is_single_transformation_valid(self, matrix, transformation, axis):
        # Validate the given transformation applies to this problem by making sure each pair
        # in the rows, columns or the diagonal are consistent
        images = self._select_images(matrix, axis, True)

        for image_1, image_2 in images:
            # If a pair doesn't match then the transformation is most likely not applicable
            if not super(_Ravens3x3Solver, self)._are_similar(transformation.apply(image_1), image_2):
                return False

        return True

    def _is_multi_transformation_valid(self, matrix, transformation, axis):
        # Validate the given transformation applies to this problem by making sure each triplet
        # in the rows, columns or the diagonal are consistent

        # For diagonal relationships, since we only have two images, a multi-transformation is not valid
        if axis == 2:
            return False

        images = self._select_images(matrix, axis, False)

        for image_1, image_2, image_3 in images:
            # If a triplet doesn't match then the transformation is most likely not applicable
            if not super(_Ravens3x3Solver, self)._are_similar(transformation.apply(image_1, other=image_2), image_3):
                return False

        return True

    def _select_images(self, matrix, axis, is_single):
        # Selects the pair of images that form the other rows, columns or diagonal
        if axis == 0:
            # Row-wise
            if is_single:
                # 'A'->'C' and 'D'->'F'
                return [(matrix[0][0], matrix[0][2]), (matrix[1][0], matrix[1][2])]
            else:
                # 'A','B'->'C' and 'D','E'->'F'
                return [(matrix[0][0], matrix[0][1], matrix[0][2]), (matrix[1][0], matrix[1][1], matrix[1][2])]
        elif axis == 1:
            # Column-wise
            if is_single:
                # 'A'->'G' and 'B'->'H'
                return [(matrix[0][0], matrix[2][0]), (matrix[0][1], matrix[2][1])]
            else:
                # 'A','D'->'G' and 'B','E'->'H'
                return [(matrix[0][0], matrix[1][0], matrix[2][0]), (matrix[0][1], matrix[1][1], matrix[2][1])]
        else:
            # Diagonal-wise, we only have the 'A'->'E'
            return [(matrix[0][0], matrix[1][1])]

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
