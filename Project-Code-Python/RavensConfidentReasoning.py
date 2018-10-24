import math

import numpy as np
from PIL import Image

from RavensFractal import RavensMutualFractalFactory


class ConfidentRavensAlgorithm:
    # TODO: Add Sphinx docs and reference to the paper

    # Dimension of the image to resize
    _RESIZE = 32
    # The representation level to use, i.e. blocks of `size x size` pixels
    _REPRESENTATION_LEVEL = 2
    # Tversky's similarity measure values that were set as described by McGreggor and Goel
    _ALPHA = 1.0
    _BETA = 1.0
    # The number of standard deviations beyond which an answer's value may be considered confident
    # In this case, 95% confidence = 1.96 standard deviations
    _CONFIDENCE = 1.96

    def __init__(self):
        self._mutual_fractal_factory = RavensMutualFractalFactory()

    def run(self, problem, problem_type):
        # The matrix order
        n = 2 if problem_type == '2x2' else 3

        # Preparatory stage
        images = self._prepare_images(problem.matrix, n)
        relationships = self._designate_relationships(images, n)

        # Execution stage
        # Represent each relationship according to the abstraction level
        self._represent_relationships(relationships)
        similarities = []

        for answer in problem.answers:
            answer = self._process_image(answer)

            if n == 2:
                theta = self._execute_2x2(answer, images, represented_relationships)
            else:  # n == 3:
                theta = self._execute_3x3(answer, images, represented_relationships)

            # Calculate a single similarity metric from the vector of values `theta`
            similarities.append(math.sqrt(np.sum(theta ** 2)))

        # Sanity check: we should not have computed more similarities that there are answers!
        assert len(similarities) == len(problem.answers), 'More similarities than answers!'

        # Evaluation
        mu = np.mean(similarities)
        sigma = np.std(similarities) / math.sqrt(len(similarities))
        deviations = (similarities - mu) / sigma

        possible_answers = np.argwhere(deviations >= self._CONFIDENCE)

        # If there is only one answer whose value is considered confident then an answer
        # can be given unambiguously and the algorithm can stop, otherwise there exists
        # ambiguity and further refinement must occur
        if len(possible_answers) == 1:
            # Answers are one-based
            # `argwhere` returns a list of lists, so we need to extract the element twice
            return possible_answers[0][0] + 1

        # If no answer has been returned, then no answer can be given unambiguously
        return None

    def _execute_2x2(self, answer, images, relationships):
        horizontal = self._mutual_fractal_factory.create([images['m3'], answer], self._REPRESENTATION_LEVEL)
        vertical = self._mutual_fractal_factory.create([images['m2'], answer], self._REPRESENTATION_LEVEL)

        return np.array([self._tversky(relationships['H1'], horizontal), self._tversky(relationships['V1'], vertical)])

    def _execute_3x3(self, answer, images, relationships):
        horizontal = self._mutual_fractal_factory.create([images['m7'], images['m8'], answer],
                                                         self._REPRESENTATION_LEVEL)
        vertical = self._mutual_fractal_factory.create([images['m3'], images['m6'], answer], self._REPRESENTATION_LEVEL)

        return np.array([
            self._tversky(relationships['H1'], horizontal),
            self._tversky(relationships['H2'], horizontal),
            self._tversky(relationships['V1'], vertical),
            self._tversky(relationships['V2'], vertical)
        ])

    def _represent_relationships(self, relationships):
        return {
            key: self._mutual_fractal_factory.create(relationship, self._REPRESENTATION_LEVEL)
            for key, relationship in relationships.iteritems()
        }

    def _prepare_images(self, matrix, n):
        if n == 2:
            return self._prepare_2x2_images(matrix)
        elif n == 3:
            return self._prepare_3x3_images(matrix)
        else:
            raise ValueError('Invalid matrix order {}!'.format(n))

    def _prepare_2x2_images(self, matrix):
        return {
            'm1': self._process_image(matrix[0][0]),
            'm2': self._process_image(matrix[0][1]),
            'm3': self._process_image(matrix[1][0])
        }

    def _prepare_3x3_images(self, matrix):
        return {
            'm1': self._process_image(matrix[0][0]),
            'm2': self._process_image(matrix[0][1]),
            'm3': self._process_image(matrix[0][2]),
            'm4': self._process_image(matrix[1][0]),
            'm5': self._process_image(matrix[1][1]),
            'm6': self._process_image(matrix[1][2]),
            'm7': self._process_image(matrix[2][0]),
            'm8': self._process_image(matrix[2][1])
        }

    def _designate_relationships(self, images, n):
        if n == 2:
            return self._designate_2x2_relationships(images)
        elif n == 3:
            return self._designate_3x3_relationships(images)
        else:
            raise ValueError('Invalid matrix order {}!'.format(n))

    def _designate_2x2_relationships(self, images):
        h1 = [images['m1'], images['m2']]
        v1 = [images['m1'], images['m3']]

        return {'H1': h1, 'V1': v1}

    def _designate_3x3_relationships(self, images):
        m1 = images['m1']
        m2 = images['m2']
        m3 = images['m3']
        m4 = images['m4']
        m5 = images['m5']
        m6 = images['m6']
        m7 = images['m7']
        m8 = images['m8']

        h1 = [m1, m2, m3]
        h2 = [m4, m5, m6]
        v1 = [m1, m4, m7]
        v2 = [m2, m5, m8]

        return {'H1': h1, 'H2': h2, 'V1': v1, 'V2': v2}

    def _tversky(self, x, y):
        # Computes the Tversky's similarity with f = number of features
        def f(z):
            return len(z)

        return f(x.intersection(y)) / (f(x.intersection(y)) + self._ALPHA * f(x - y) + self._BETA * f(y - x))

    def _process_image(self, image):
        # Resize the image and "binarize" it to avoid dealing with different tones of gray
        # The threshold was found after some empirical experimentation
        return np.array(
            image.resize((self._RESIZE, self._RESIZE), resample=Image.BICUBIC).point(lambda x: 255 if x > 200 else 0)
        )
