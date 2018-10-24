from collections import namedtuple

import numpy as np
from PIL import Image

_Transformation = namedtuple('Transformation', ['name', 'transform'])
_Analogy = namedtuple('Analogy', ['type', 'left', 'right', 'transformations'])
_Operand = namedtuple('Operand', ['operation', 'value'])

# Types of analogies
_UNARY = 0
_BINARY = 1

# Types of image composition operands
_ADDITION = 0
_SUBTRACTION = 1

# Size to resize images
_RESIZE = 32


class RavensAffine3x3Solver:
    # TODO: Add Sphinx docs and references to papers

    def __init__(self):
        pass

    def run(self, problem):
        images = self._prepare_images(problem.matrix)
        finder = _AffineBestFitTransformationFinder(images)

        analogy, transformation, (tx, ty), operand = finder.find()

        # Apply transformation
        guess = (transformation.transform(analogy.right[0]) if analogy.type == _UNARY
                 else transformation.transform(analogy.right[0], other=analogy.right[1]))

        # Apply translation
        guess = np.roll(guess, tx, axis=1)
        guess = np.roll(guess, ty, axis=0)

        # Apply operand, if any
        if operand:
            if operand.operation == _ADDITION:
                guess += operand.value
            else:
                guess -= operand.value

        # Select the best answer
        similarities = [
            _tversky(guess, self._process_image(answer), 1.0, 1.0)
            for answer in problem.answers
        ]

        best_answer = np.argmax(similarities)

        # Answers are 1-indexed based
        return best_answer + 1

    def _prepare_images(self, matrix):
        return {
            'A': self._process_image(matrix[0][0]),
            'B': self._process_image(matrix[0][1]),
            'C': self._process_image(matrix[0][2]),
            'D': self._process_image(matrix[1][0]),
            'E': self._process_image(matrix[1][1]),
            'F': self._process_image(matrix[1][2]),
            'G': self._process_image(matrix[2][0]),
            'H': self._process_image(matrix[2][1])
        }

    def _process_image(self, image):
        return np.array(image.resize((_RESIZE, _RESIZE), resample=Image.BICUBIC).point(lambda x: 0 if x > 200 else 1))


class _AffineBestFitTransformationFinder:
    def __init__(self, images):
        self._unary_transformations = [
            _Transformation(name='Identity', transform=lambda x: x),
            _Transformation(name='Horizontal Reflection', transform=lambda x, **kwargs: np.fliplr(x)),
            _Transformation(name='Vertical Reflection', transform=lambda x, **kwargs: np.flipud(x)),
            _Transformation(name='Rotation 90', transform=lambda x, **kwargs: np.rot90(x)),
            _Transformation(name='Rotation 180', transform=lambda x, **kwargs: np.rot90(x, k=2)),
            _Transformation(name='Rotation 270', transform=lambda x, **kwargs: np.rot90(x, k=3))
        ]

        self._binary_transformations = [
            _Transformation(name='Union', transform=lambda x, **kwargs: np.maximum(x, kwargs['other'])),
            _Transformation(name='Intersection', transform=lambda x, **kwargs: np.minimum(x, kwargs['other'])),
            _Transformation(name='Subtraction', transform=lambda x, **kwargs: x - kwargs['other']),
            _Transformation(name='Back-subtraction', transform=lambda x, **kwargs: kwargs['other'] - x),
            _Transformation(name='XOR', transform=lambda x, **kwargs: np.maximum(x, kwargs['other']) -
                                                                      np.minimum(x, kwargs['other']))
        ]

        self._analogies = [
            # Unary analogies
            _Analogy(type=_UNARY, left=[images['A'], images['B']], right=[images['H']],
                     transformations=self._unary_transformations),
            _Analogy(type=_UNARY, left=[images['B'], images['C']], right=[images['H']],
                     transformations=self._unary_transformations),
            _Analogy(type=_UNARY, left=[images['D'], images['E']], right=[images['H']],
                     transformations=self._unary_transformations),
            _Analogy(type=_UNARY, left=[images['E'], images['F']], right=[images['H']],
                     transformations=self._unary_transformations),
            _Analogy(type=_UNARY, left=[images['G'], images['H']], right=[images['H']],
                     transformations=self._unary_transformations),
            _Analogy(type=_UNARY, left=[images['A'], images['C']], right=[images['G']],
                     transformations=self._unary_transformations),
            _Analogy(type=_UNARY, left=[images['D'], images['F']], right=[images['G']],
                     transformations=self._unary_transformations),
            _Analogy(type=_UNARY, left=[images['A'], images['D']], right=[images['F']],
                     transformations=self._unary_transformations),
            _Analogy(type=_UNARY, left=[images['D'], images['G']], right=[images['F']],
                     transformations=self._unary_transformations),
            _Analogy(type=_UNARY, left=[images['B'], images['E']], right=[images['F']],
                     transformations=self._unary_transformations),
            _Analogy(type=_UNARY, left=[images['E'], images['H']], right=[images['F']],
                     transformations=self._unary_transformations),
            _Analogy(type=_UNARY, left=[images['C'], images['F']], right=[images['F']],
                     transformations=self._unary_transformations),
            _Analogy(type=_UNARY, left=[images['A'], images['G']], right=[images['C']],
                     transformations=self._unary_transformations),
            _Analogy(type=_UNARY, left=[images['B'], images['H']], right=[images['C']],
                     transformations=self._unary_transformations),
            # Binary analogies
            _Analogy(type=_BINARY, left=[images['A'], images['B'], images['C']], right=[images['G'], images['H']],
                     transformations=self._binary_transformations),
            _Analogy(type=_BINARY, left=[images['D'], images['E'], images['F']], right=[images['G'], images['H']],
                     transformations=self._binary_transformations),
            _Analogy(type=_BINARY, left=[images['A'], images['D'], images['G']], right=[images['C'], images['F']],
                     transformations=self._binary_transformations),
            _Analogy(type=_BINARY, left=[images['B'], images['E'], images['H']], right=[images['C'], images['F']],
                     transformations=self._binary_transformations)
        ]

    def find(self):
        best_similarity = np.float('-inf')
        best_fit = None

        for i, analogy in enumerate(self._analogies):
            for transformation in analogy.transformations:
                transformed = (transformation.transform(analogy.left[0]) if analogy.type == _UNARY
                               else transformation.transform(analogy.left[0], other=analogy.left[1]))

                # Find x-axis translation
                tx = self._find_best_translation(transformed, analogy.left[-1], 1)
                # Find y-axis translation
                ty = self._find_best_translation(transformed, analogy.left[-1], 0)

                similarity, operand = self._find_image_composition_operand(transformed, analogy)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_fit = (analogy, transformation, (tx, ty), operand)

        return best_fit

    def _find_best_translation(self, transformed, image, axis):
        translations = np.array([
            _tversky(np.roll(transformed, delta, axis=axis), image, 1.0, 1.0)
            for delta in range(0, transformed.shape[0])
        ])

        return np.argmax(translations)

    def _find_image_composition_operand(self, transformed, analogy):
        similarities = [
            _tversky(transformed, analogy.left[-1], 1.0, 1.0),
            _tversky(transformed, analogy.left[-1], 1.0, 0.0),
            _tversky(transformed, analogy.left[-1], 0.0, 1.0)
        ]

        smax = np.argmax(similarities)

        if smax == 0:
            # Case 1
            return similarities[smax], None
        elif smax == 1:
            # Case 2
            value = (analogy.left[1] - analogy.left[0] if analogy.type == _UNARY
                     else analogy.left[2] - analogy.left[1] - analogy.left[0])

            return similarities[smax], _Operand(operation=_ADDITION, value=value)
        else:
            # Case 3
            value = (analogy.left[0] - analogy.left[1] if analogy.type == _UNARY
                     else analogy.left[0] - analogy.left[1] - analogy.left[2])

            return similarities[smax], _Operand(operation=_SUBTRACTION, value=value)


def _tversky(a, b, alpha, beta):
    # Summation of feature comparison values
    def f(x):
        return float(np.sum(x))

    def intersection(x, y):
        return np.minimum(x, y)

    def union(x, y):
        return np.maximum(x, y)

    if alpha == 1 and beta == 1:
        return f(intersection(a, b)) / f(union(a, b))

    a_intersection_b = f(intersection(a, b))

    return a_intersection_b / (a_intersection_b + alpha * f(a - b) + beta * f(b - a))
