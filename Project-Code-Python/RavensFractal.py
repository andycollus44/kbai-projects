from collections import namedtuple

import numpy as np

_FractalTransformation = namedtuple('FractalTransformation', ['name', 'transform'])
_FractalCandidate = namedtuple('FractalCandidate', ['rmse', 'fractal'])


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


class RavensFractalEncoder:
    """
    Implements the fractal encoding algorithm as defined by McGreggor, Kunda and Goel in
    "A Fractal Approach Towards Visual Analogy".

    Reference: http://dilab.gatech.edu/publications/McGreggorKundaGoel_ICCCX_2010.pdf
    """

    def __init__(self):
        # The list of affine transformations to apply
        self._transformations = [
            _FractalTransformation(name='Identity', transform=lambda x: x),
            _FractalTransformation(name='Horizontal Reflection', transform=lambda x: np.fliplr(x)),
            _FractalTransformation(name='Vertical Reflection', transform=lambda x: np.flipud(x)),
            _FractalTransformation(name='Rotation 90', transform=lambda x: np.rot90(x)),
            _FractalTransformation(name='Rotation 180', transform=lambda x: np.rot90(x, k=2)),
            _FractalTransformation(name='Rotation 270', transform=lambda x: np.rot90(x, k=3)),
            _FractalTransformation(name='Color Shift', transform=lambda x: 255 - x)
        ]

    def apply(self, source, destination, size):
        """
        Applies the fractal encoding algorithm to the given source and destination images.

        :param source: The source image.
        :type source: ndarray
        :param destination: The destination image.
        :type destination: ndarray
        :param size: The size of the representation.
        :type size: int
        :return: A set of fractal features derived from the fractal encoding.
        :rtype: set
        """
        assert source.shape == destination.shape, 'Both source and destination must have the same shape!'

        # Decompose the destination and source into a set of N smaller images of size `size x size`
        N = destination.shape[0] // size
        destination_index = [(col * size, row * size) for row in range(0, N) for col in range(0, N)]
        source_index = destination_index[:]

        fractals = []

        # For each block in the destination
        for d_index in destination_index:
            # Retrieve the corresponding block from the destination image
            destination_image = destination[d_index[0]:d_index[0] + size, d_index[1]:d_index[1] + size]
            # Find the best candidate that approaches the destination image given the source image
            best_candidate = self._find_best_candidate(destination_image, d_index, source, source_index, size)

            fractals.append(best_candidate.fractal)

        # Make sure the destination image was encoded correctly.
        # The number of fractals should be the same as the number of blocks the destination image was decomposed into
        assert len(fractals) == len(destination_index), 'Destination image was not decomposed correctly!'

        return self._to_features(fractals)

    def _find_best_candidate(self, destination_image, destination_image_index, source, source_index, size):
        # Examine the entire source image for an equivalent block such that an affine transformation
        # will result in the destination block
        best_candidate = None

        for s_index in source_index:
            # Retrieve the corresponding block from the source image
            source_image = source[s_index[0]:s_index[0] + size, s_index[1]:s_index[1] + size]

            for transformation in self._transformations:
                transformed = transformation.transform(source_image)
                rmse = self._similarity(destination_image.astype(int), transformed.astype(int))

                if not best_candidate or rmse > best_candidate.rmse:
                    # Only keep the best candidate

                    # The location of the leftmost and topmost pixel in the source image
                    sx, sy = s_index
                    # The location of the leftmost and topmost pixel in the destination image
                    dx, dy = destination_image_index
                    # The affine transformation to be used
                    k = transformation.name
                    # Whether the color of the image was inverted or not
                    c = transformation.name == 'Color Shift'

                    best_candidate = _FractalCandidate(rmse=rmse, fractal=((sx, sy), (dx, dy), k, c))

        return best_candidate

    def _to_features(self, fractals):
        # Generate the set of features given a list of fractals
        features = set()

        for fractal in fractals:
            source_position, destination_position, transformation, color_shift = fractal

            # A specific feature
            features.add(fractal)
            # A position agnostic feature
            features.add(((source_position[0] - destination_position[0], source_position[1] - destination_position[1]),
                          transformation, color_shift))
            # An affine transformation agnostic feature
            features.add((source_position, destination_position, color_shift))
            # A color agnostic feature
            features.add((source_position, destination_position, transformation))
            # An affine specific feature
            features.add((transformation, color_shift)),
            # A color shift specific feature
            features.add(color_shift)

        return features

    def _similarity(self, image, other_image):
        # Computes the similarity between this image and another one using the Normalized Root Mean Squared Error
        # References:
        # - https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
        # - https://tinyurl.com/ybmwlsur
        rmse = np.sqrt(np.mean((image - other_image) ** 2))
        max_val = max(np.max(image), np.max(other_image))
        min_val = min(np.min(image), np.min(other_image))

        # `max_val - min_val == 0` happens when the two images are either all black or all white
        return 1 - rmse if max_val - min_val == 0 else 1 - (rmse / (max_val - min_val))
