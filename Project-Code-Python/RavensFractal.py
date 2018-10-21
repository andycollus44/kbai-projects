import time
from collections import namedtuple

import numpy as np
from numpy.lib import stride_tricks

_FractalTransformation = namedtuple('FractalTransformation', ['name', 'transform'])
_FractalCandidate = namedtuple('FractalCandidate', ['correspondence', 'fractal'])


class RavensMutualFractalFactory:
    """
    A factory for creating MutualFractal objects.
    """
    def __init__(self):
        self._encoder = RavensFractalEncoder()

    def create(self, images, size):
        """
        Creates a MutualFactory object given some images and the size of the representation.

        :param images: A list of images represented as arrays of pixels.
        :type images: list[ndarray]
        :param size: The size of the representation.
        :type size: int
        :return: A MutualFractal object.
        :rtype: MutualFractal
        """
        if len(images) == 2:
            fractals_ab = self._encoder.apply(images[0], images[1], size)
            fractals_ba = self._encoder.apply(images[1], images[0], size)

            fractals = [fractals_ab, fractals_ba]
        elif len(images) == 3:
            # TODO: Fix implementation (see notes in phone)!
            fractals_ij = self._encoder.apply(images[0], images[1], size)
            fractals_jk = self._encoder.apply(images[1], images[2], size)
            fractals_ik = self._encoder.apply(images[0], images[2], size)

            fractals = [fractals_ij, fractals_jk, fractals_ik]
        else:
            raise ValueError('Cannot create MutualFractal for {} images!'.format(len(images)))

        return RavensMutualFractal(fractals)


class RavensMutualFractal:
    """
    Represents a mutual fractal as defined by McGreggor and Goel in "Fractal Analogies for General Intelligence".

    It only support representations of pairs and triplets.
    Reference: http://dilab.gatech.edu/publications/McGreggor%20Goel%202012%20AGI.pdf
    """
    def __init__(self, fractals):
        assert 2 <= len(fractals) <= 3, 'Only pairs and triplets are supported!'

        if len(fractals) == 2:
            self._features = fractals[0].union(fractals[1])
        elif len(fractals) == 3:
            self._features = fractals[0].union(fractals[1]).union(fractals[2])

    @property
    def features(self):
        return self._features

    def intersection(self, other):
        """
        Intersects this MutualFractal's set of features with another MutualFractal.

        :param other: The other MutualFractal.
        :type other: RavensMutualFractal
        :return: Intersection between the features.
        :rtype: set
        """
        return self._features.intersection(other.features)

    def __sub__(self, other):
        """
        Computes the difference between this MutualFractal's features and another MutualFractal.

        :param other: The other MutualFractal.
        :type other: RavensMutualFractal
        :return: The difference between the features.
        :rtype: set
        """
        return self._features - other.features


class RavensFractalEncoder:
    """
    Implements the fractal encoding algorithm as defined by McGreggor, Kunda and Goel in
    "A Fractal Approach Towards Visual Analogy", and as described by McGreggor in his
    doctoral dissertation "Fractal Reasoning".

    This implementation contains some optimization tricks to speed up its computation.

    References:
        - http://dilab.gatech.edu/publications/McGreggorKundaGoel_ICCCX_2010.pdf
        - https://smartech.gatech.edu/bitstream/handle/1853/50337/MCGREGGOR-DISSERTATION-2013.pdf
    """

    # The weights to use in the correspondence computation chosen after empirical experimentation
    _PHOTOMETRIC_WEIGHT = 0.8
    _DISTANCE_WEIGHT = 0.2

    def __init__(self):
        # The list of affine transformations to apply as described in McGreggor's dissertation
        # This transformations are vectorized to work with a stack of `size x size` blocks
        # Reference:
        # https://stackoverflow.com/a/43864937
        self._transformations = [
            _FractalTransformation(name='Identity', transform=lambda x: x),
            _FractalTransformation(name='Horizontal Reflection', transform=lambda x: x[..., ::-1]),
            _FractalTransformation(name='Vertical Reflection', transform=lambda x: x[..., ::-1, :]),
            _FractalTransformation(name='Rotation 90', transform=lambda x: x.swapaxes(-2, -1)[..., ::-1, :]),
            _FractalTransformation(name='Rotation 180', transform=lambda x: x[..., ::-1, ::-1]),
            _FractalTransformation(name='Rotation 270', transform=lambda x: x.swapaxes(-2, -1)[..., ::-1]),
            _FractalTransformation(name='Reflection YnX', transform=lambda x: x.swapaxes(-2, -1)[..., ::-1, ::-1]),
            _FractalTransformation(name='Reflection YX', transform=lambda x: x.swapaxes(-2, -1))
        ]

        # A cache for candidate fractals
        self._cached_candidates = {}

    def apply(self, source, destination, size):
        """
        Applies the fractal encoding algorithm to the given source and destination images.

        :param source: The source image.
        :type source: ndarray
        :param destination: The destination image.
        :type destination: ndarray
        :param size: The size of the representation.
        :type size: int
        :return: A tuple with the list of fractals and a set of fractal features derived from the fractal encoding.
        :rtype: tuple
        """
        assert source.shape == destination.shape, 'Both source and destination must have the same shape!'

        # The cache is only valid per run
        self._cached_candidates.clear()

        # Decompose the destination and source into a set of N smaller images of size `size x size`
        # This will be done in a sliding window fashion with a step size, or stride, of 1 and blocks of `size x size`
        N = destination.shape[0] - size + 1
        block_indices = [(row, col) for row in range(0, N) for col in range(0, N)]

        source_blocks = self._decompose(source, size)
        destination_blocks = self._decompose(destination, size)

        # For each block in the destination, find the best candidate that approaches the destination given the source
        # As an optimization, skip blocks that are completely white as we will assume the image always has a white
        # background, and thus we will only be encoding the black portions which correspond to shapes
        fractals = [
            self._find_best_candidate(destination_blocks[i], i, source_blocks, block_indices).fractal
            for i in range(0, len(block_indices))
            if not np.alltrue(destination_blocks[i] == 255)
        ]

        return fractals, self._to_features(fractals)

    def _find_best_candidate(self, destination_block, destination_index, source_blocks, block_indices):
        # Examine the entire source image for an equivalent block such that an affine transformation
        # will result in the destination block

        cached_candidate = self._get_cached_candidate(destination_block)

        if cached_candidate is not None:
            return _FractalCandidate(correspondence=cached_candidate[0], fractal=(cached_candidate[1],
                                                                                  block_indices[destination_index],
                                                                                  cached_candidate[2]))

        best_candidate = None
        best_found = False

        for transformation in self._transformations:
            # Transform all blocks at the same time using vectorization
            transformed_blocks = transformation.transform(source_blocks)

            for s_index, s_block in enumerate(transformed_blocks):
                correspondence = self._correspondence(destination_block, s_block)

                if not best_candidate or correspondence < best_candidate.correspondence:
                    # Only keep the best candidate
                    best_candidate = _FractalCandidate(correspondence=correspondence,
                                                       fractal=(block_indices[s_index],
                                                                block_indices[destination_index],
                                                                transformation.name))

                    # Optimization - If the correspondence is 0, e.g. all black blocks, stop the search/return
                    if int(correspondence) == 0:
                        best_found = True
                        break

            if best_found:
                break

        self._cache_candidate(destination_block, best_candidate)

        return best_candidate

    def _get_cached_candidate(self, block):
        # Use the string representation of the Numpy array as the key
        # Reference: https://stackoverflow.com/a/16592241
        key = hash(block.tostring())

        if key in self._cached_candidates:
            return self._cached_candidates[key]

        return None

    def _cache_candidate(self, block, candidate):
        s_block, _, transformation = candidate.fractal
        self._cached_candidates[hash(block.tostring())] = (candidate.correspondence, s_block, transformation)

    def _to_features(self, fractals):
        # Generate the set of features given a list of fractals
        features = set()

        for fractal in fractals:
            source_position, destination_position, transformation = fractal

            features.update([
                # A specific feature
                fractal,
                # A position agnostic feature
                ((source_position[0] - destination_position[0], source_position[1] - destination_position[1]),
                 transformation),
                # An affine transformation agnostic feature
                (source_position, destination_position),
                # A color agnostic feature
                (source_position, destination_position, transformation),
                # An affine specific feature
                transformation
            ])

        return features

    def _decompose(self, array, size):
        # Generates a list of `size x size` blocks following a sliding window with stride 1
        # Reference: https://realpython.com/numpy-array-programming/#image-feature-extraction
        return stride_tricks.as_strided(
            array,
            shape=(array.shape[0] - size + 1, array.shape[1] - size + 1, size, size),
            strides=2 * array.strides
        ).reshape(-1, size, size)

    def _correspondence(self, image, other_image):
        # Computes the correspondence between `image` and `other_image`
        # Photometric correspondence, i.e. square error and distance
        # For distance, see:
        # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
        return (self._PHOTOMETRIC_WEIGHT * np.sum((other_image - image) ** 2) +
                self._DISTANCE_WEIGHT * np.linalg.norm(other_image - image))
