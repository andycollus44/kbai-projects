from abc import ABCMeta, abstractmethod, abstractproperty
from collections import namedtuple
from operator import itemgetter

import numpy as np
from PIL import Image

from RavensTransformation import (SINGLE, MULTI, FlipTransformation, MirrorTransformation, NoOpTransformation,
                                  ShapeFillTransformation, XORTransformation)

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
            raise ValueError('3x3 problems are not supported!')
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
        # Apply each transformation for each axis (row=0, column=1)
        answers = [self._apply(problem, transformation, axis) for axis in [0, 1] for transformation in self._transforms]
        # Sort them in increasing order by their similarity measure
        answers.sort(key=itemgetter(0), reverse=True)

        # The best answer is the first sorted element
        _, best = answers[0]

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

    def _resize(self, image):
        # Reduce size of the image to 32 x 32 so that operations are faster, and similarities are easier to find
        return image.resize((32, 32), resample=Image.BICUBIC)

    @abstractproperty
    def _transforms(self):
        # The list of all available transformations for this solver
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
            XORTransformation()
        ]

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
        expected = transformation.apply(image_1, **{'other': image_2})

        # The 'C' image for row-wise or the 'B' image for column-wise
        last_image = matrix[1][0] if axis == 0 else matrix[0][1]
        # For each answer, apply the multi-transformation with the last image to generate potential candidates
        # and test its similarity with the expected result, i.e. generate and test
        similarities = [
            super(_Ravens2x2Solver, self)._similarity(expected, transformation.apply(last_image, **{'other': answer}))
            for answer in answers
        ]

        # In contrast with the single-transformations, for multi ones, there is no initial check between the first
        # two pairs to validate they are similar since the two images are needed to obtain a transformation and this
        # result is compared to the application of the same transformation against the last image with the answers
        # So, to avoid answering problems incorrectly, filter out the answers that are not significantly similar
        filtered = filter(lambda similarity: similarity >= super(_Ravens2x2Solver, self)._SIMILARITY_THRESHOLD,
                          similarities)

        # If there are no significantly similar candidates, then there is no answer
        if len(filtered) == 0:
            return 0., None

        # Find the best candidate based on similarity
        candidate = np.argmax(similarities)

        return similarities[candidate], candidate
