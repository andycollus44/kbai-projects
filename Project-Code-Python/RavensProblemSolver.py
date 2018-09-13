from abc import ABCMeta, abstractmethod, abstractproperty
from collections import namedtuple
from operator import itemgetter

import numpy as np

from RavensTransformation import FlipTransformation, MirrorTransformation, NoOpTransformation

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

        # If no answer was found by any of the transformations, then skip the problem (-1), else return the best answer
        return -1 if best is None else best

    def _are_similar(self, image, other_image):
        return self._similarity(image, other_image) >= self._SIMILARITY_THRESHOLD

    def _similarity(self, image, other_image):
        # Computes the similarity between this image and another one using the Normalized Root Mean Squared Error
        # References:
        # - https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
        # - https://tinyurl.com/ybmwlsur
        image = np.array(image).astype(np.int)
        other_image = np.array(other_image).astype(np.int)

        rmse = np.sqrt(np.mean((image - other_image) ** 2))
        max_val = max(np.max(image), np.max(other_image))
        min_val = min(np.min(image), np.min(other_image))

        return 1 - (rmse / (max_val - min_val))

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
            FlipTransformation()
        ]

    def _apply(self, problem, transformation, axis):
        matrix = problem.matrix
        answers = problem.answers

        # The 'A' image
        image_1 = transformation.apply(matrix[0][0])
        # Either the 'B' image for row-wise or the 'C' image for column-wise
        image_2 = matrix[0][1] if axis == 0 else matrix[1][0]

        # Compare the image with its pair, if they don't match then their is no answer
        if not super(_Ravens2x2Solver, self)._are_similar(image_1, image_2):
            return Answer(similarity=0., answer=None)

        # Find the answer that most closely matches the transformed image, i.e. generate and test
        # The 'C' image for row-wise or the 'B' image for column-wise
        last_image = transformation.apply(matrix[1][0]) if axis == 0 else transformation.apply(matrix[0][1])
        similarities = [super(_Ravens2x2Solver, self)._similarity(last_image, answer) for answer in answers]
        candidate = np.argmax(similarities)

        # Answers are one-indexed based
        return Answer(similarity=similarities[candidate], answer=candidate + 1)
