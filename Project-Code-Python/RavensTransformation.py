from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from PIL import ImageChops, ImageFilter, ImageOps, Image, ImageDraw

from RavensShape import RavensShapeExtractor

# Types of transformations with respect to the images they act on
SINGLE = 'SINGLE'
MULTI = 'MULTI'

# Keep singleton instances available to all semantic relationships
_extractor = RavensShapeExtractor()


class Transformation:
    __metaclass__ = ABCMeta

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def type(self):
        pass

    @abstractproperty
    def confidence(self):
        """
        Allows underlying transformations to override the default confidence threshold.

        :return: The confidence threshold to use for this particular transformation.
        """
        pass

    @abstractmethod
    def apply(self, image, **kwargs):
        """
        Applies this transformation to the given image.

        :param image: The image to apply this transformation to.
        :type image: PIL.Image.Image
        :param kwargs: Extra arguments for this transformation, e.g. `other` for multi-transformations.
        :type kwargs: dict
        :return: The transformed image.
        :rtype: PIL.Image.Image
        """
        pass

    def __repr__(self):
        return self.name


class SingleTransformation(Transformation):
    __metaclass__ = ABCMeta

    @property
    def type(self):
        return SINGLE

    @property
    def confidence(self):
        return None


class MultiTransformation(Transformation):
    __metaclass__ = ABCMeta

    @property
    def type(self):
        return MULTI

    @property
    def confidence(self):
        return None

    def _validate(self, **kwargs):
        if not kwargs.get('other', None):
            raise ValueError('Transformations of type {} require parameter `other`'.format(self.type))


class NoOpTransformation(SingleTransformation):
    """
    A no-op transformation returns a copy of the given image.
    """
    @property
    def name(self):
        return 'NoOp'

    def apply(self, image, **kwargs):
        return image.copy()


class MirrorTransformation(SingleTransformation):
    """
    A mirror transformation flips the image horizontally (left to right).
    """
    @property
    def name(self):
        return 'Mirror'

    def apply(self, image, **kwargs):
        return ImageOps.mirror(image)


class FlipTransformation(SingleTransformation):
    """
    A flip transformation flips the image vertically (top to bottom).
    """
    @property
    def name(self):
        return 'Flip'

    def apply(self, image, **kwargs):
        return ImageOps.flip(image)


class ShapeFillTransformation(SingleTransformation):
    """
    A shape fill transformation looks at the shapes inside an image and fills them in.
    """
    def __init__(self):
        self._shape_extractor = RavensShapeExtractor()

    @property
    def name(self):
        return 'ShapeFill'

    def apply(self, image, **kwargs):
        # Extract shapes from the image
        shapes = self._shape_extractor.apply(image)

        # For each shape, reconstruct the polygons and fill them
        reconstructed = Image.new('L', image.size, 255)
        draw = ImageDraw.Draw(reconstructed)

        for shape in shapes:
            # The `polygon` function expects a flattened list of continuous x,y pairs
            draw.polygon(shape.contour.flatten().tolist(), fill=0, outline=255)

        return reconstructed


class RotationTransformation(SingleTransformation):
    """
    A rotation transformation rotates the image by certain degrees clockwise.
    """

    def __init__(self, degrees):
        self._degrees = degrees

    @property
    def name(self):
        return 'Rotation'

    def apply(self, image, **kwargs):
        # Image.rotate() rotates the image counterclockwise,
        # use the negative to rotate the image clockwise
        return image.rotate(-self._degrees, resample=Image.BICUBIC)


class ImageDuplication(SingleTransformation):
    """
    An image duplication transformation clones the given image the specified number of times along a particular axis.
    """
    # Axes for duplication
    HORIZONTAL = 0
    VERTICAL = 1
    DIAGONAL = 2
    INVERTED_DIAGONAL = 3

    # Duplication modes
    OVERLAPPING = 0
    NON_OVERLAPPING = 1
    SIDE_BY_SIDE = 2

    # The frame being produced with respect to the matrix
    MIDDLE_FRAME = 0
    LAST_FRAME = 1

    # Helpful aliases for the parameters of this transformation
    TWO_TIMES_MIDDLE_FRAME_THREE_TIMES_LAST_FRAME = {
        MIDDLE_FRAME: 2,
        LAST_FRAME: 3
    }

    TWO_TIMES_ALL_FRAMES = {
        MIDDLE_FRAME: 2,
        LAST_FRAME: 2
    }

    ALL_FRAMES_OVERLAPPING = {
        MIDDLE_FRAME: OVERLAPPING,
        LAST_FRAME: OVERLAPPING
    }

    All_FRAMES_NON_OVERLAPPING = {
        MIDDLE_FRAME: NON_OVERLAPPING,
        LAST_FRAME: NON_OVERLAPPING
    }

    ALL_FRAMES_SIDE_BY_SIDE = {
        MIDDLE_FRAME: SIDE_BY_SIDE,
        LAST_FRAME: SIDE_BY_SIDE
    }

    MIDDLE_FRAME_OVERLAPPING_LAST_FRAME_NON_OVERLAPPING = {
        MIDDLE_FRAME: OVERLAPPING,
        LAST_FRAME: NON_OVERLAPPING
    }

    def __init__(self, times_per_frame, mode_per_frame, axis):
        """
        :param times_per_frame: The number of times to duplicate the image per frame.
        :type times_per_frame: dict
        :param mode_per_frame: The mode to use for duplication per frame.
                               One of `OVERLAPPING` or `NON_OVERLAPPING` or `SIDE_BY_SIDE`.
        :type mode_per_frame: dict
        :param axis: The axis to use for duplication. One of `HORIZONTAL`, `VERTICAL` or `DIAGONAL`.
        """
        self._times_per_frame = times_per_frame
        self._mode_per_frame = mode_per_frame
        self._axis = axis

    @property
    def name(self):
        return 'ImageDuplication'

    @property
    def confidence(self):
        # Override the confidence threshold because the images will not match perfectly based on
        # the duplication operators; however, they should be fairly similar, but not as strict
        # as the default 90% confidence threshold
        return 0.8

    def apply(self, image, **kwargs):
        if kwargs.get('A', None) is None:
            raise ValueError('Transformation {} requires parameters `A`'.format(self.name))

        # Default to the last frame if no frame is passed
        # This is to support applying this transformation to the middle frames, if needed
        frame = kwargs.get('frame', self.LAST_FRAME)

        # Extract the shape from image 'A'
        shapes = _extractor.apply(kwargs['A'])

        # For this transformation, we assume image 'A' always has a single shape
        # if that is not the case, then we cannot apply this transformation so
        # we return a black image which most likely will not match with anything
        if len(shapes) != 1:
            return Image.new(image.mode, image.size)

        width, height = shapes[0].width, shapes[0].height

        result = None

        if self._mode_per_frame[frame] == self.OVERLAPPING:
            if self._axis == self.HORIZONTAL:
                # Use a third the width of the shape as the duplication offset in order for there to be some overlap
                result = self._duplicate(image.copy(), frame, int(width / 3.0), self._move_horizontally)
            elif self._axis == self.VERTICAL:
                # Use a third the height of the shape as the duplication offset in order for there to be some overlap
                result = self._duplicate(image.copy(), frame, int(height / 3.0), self._move_vertically)
        elif self._mode_per_frame[frame] == self.NON_OVERLAPPING:
            if self._axis == self.HORIZONTAL:
                # Use the whole width plus a little bit more (a third of the width) of the shape as the offset
                # to avoid overlap horizontally
                result = self._duplicate(image.copy(), frame, width + int(width / 3.0), self._move_horizontally)
            elif self._axis == self.VERTICAL:
                # Use the whole height plus a little bit more (a third of the height) of the shape as the offset
                # to avoid overlap vertically
                result = self._duplicate(image.copy(), frame, height + int(height / 3.0), self._move_vertically)
            elif self._axis == self.DIAGONAL:
                # Use the whole width of the shape plus half the height as the offset
                # to avoid overlap both vertically and horizontally
                result = self._duplicate(image.copy(), frame, int(width + height / 2.0), self._move_diagonally)
            elif self._axis == self.INVERTED_DIAGONAL:
                # Use the whole width of the shape plus half the height as the offset
                # to avoid overlap both vertically and horizontally
                result = self._duplicate(image.copy(), frame, int(width + height / 2.0), self._move_diagonally_inverted)
        elif self._mode_per_frame[frame] == self.SIDE_BY_SIDE:
            if self._axis == self.HORIZONTAL:
                # Use the whole width  of the shape as the offset to places images side by side
                result = self._duplicate(image.copy(), frame, width, self._move_horizontally)
            elif self._axis == self.VERTICAL:
                # Use the whole height  of the shape as the offset to places images side by side
                result = self._duplicate(image.copy(), frame, width, self._move_vertically)
        else:
            raise ValueError('Invalid mode {}!'.format(self._mode_per_frame[frame]))

        # Reconstruct the image back to its Pillow representation
        return Image.fromarray(result)

    def _duplicate(self, image, frame, offset, duplication_operator):
        times = self._times_per_frame[frame]

        # Generate `1 - times` duplicates alternating between left and right
        # We generate `1 -times` because the original image is one of said duplicates!
        duplicates = []
        direction = 1
        offset_increment = 1

        for t in range(0, times - 1):
            duplicates.append(duplication_operator(image, offset * offset_increment * direction))
            direction *= -1

            # Every two duplicates we need to double our increment to move the duplicate further
            if t > 0 and t % 2 == 0:
                offset_increment *= 2

        # Merge all those duplicates
        unified = self._union(image, duplicates)

        # Center the image if the number of duplicates is even by moving it back by half the offset
        if times % 2 == 0:
            unified = duplication_operator(unified, -offset / 2)

        return unified

    def _move_horizontally(self, image, offset):
        # Move the whole image horizontally by "rolling" the matrix
        return np.roll(image, offset, 1)

    def _move_vertically(self, image, offset):
        # Move the whole image vertically by "rolling" the matrix
        return np.roll(image, offset, 0)

    def _move_diagonally(self, image, offset):
        # Move the whole image diagonally by "rolling" the matrix vertically and horizontally
        return self._move_horizontally(self._move_vertically(image, offset), offset)

    def _move_diagonally_inverted(self, image, offset):
        # Move the whole image diagonally by "rolling" the matrix vertically and horizontally
        # one movement to one side and another movement to the opposite side effectively being inverted
        return self._move_horizontally(self._move_vertically(image, offset), -offset)

    def _union(self, image, duplicates):
        # The union operation as defined by Kunda in his doctoral dissertation
        # Reference: https://smartech.gatech.edu/bitstream/handle/1853/47639/kunda_maithilee_201305_phd.pdf
        # Here we use minimum instead of maximum because Kunda assumed that the images had a value of 0 for white
        # but, in reality, 0 indicates a black pixel and 255 (or 1 if the image is binary) is white
        unified = np.minimum(image, duplicates[0])

        for i in range(1, len(duplicates)):
            unified = np.minimum(unified, duplicates[i])

        return unified


class ImageSwitchSidesHorizontallyTransformation(SingleTransformation):
    """
    An image switch transformation switches two sides of an image horizontally.
    """
    @property
    def name(self):
        return 'ImageSwitchSidesHorizontally'

    @property
    def confidence(self):
        # Override the confidence threshold because the images will not match perfectly based on
        # the movement operators; however, they should be fairly similar, but not as strict
        # as the default 90% confidence threshold
        return 0.8

    def apply(self, image, **kwargs):
        # For this transformation, we assume only two shapes are present and each one is exactly
        # at each half of the image, horizontally

        im = np.array(image.copy())
        half = image.width / 2

        # Partition the image into a left and right side
        # In an image, the 'y' column is the row so we slice using the y-axis
        left = im[:, 0:half]
        right = im[:, half:]

        # Switch sides
        switched = np.hstack((right, left))

        # Reconstruct the image back to its Pillow representation
        return Image.fromarray(switched)


class ImageSegmentTopDownDeletion(SingleTransformation):
    """
    An image segment top-down deletion transformation removes the specified number of segments from a partitioned image.
    """
    def __init__(self, segments, to_delete):
        self._segments = segments
        self._to_delete = to_delete

    @property
    def name(self):
        return 'ImageSegmentTopDownDeletion'

    def apply(self, image, **kwargs):
        im = np.array(image.copy())

        # Segment the image according to the given number of segments
        partition = image.height / self._segments
        segmented = [
            im[partition * i:partition * (i + 1)]
            for i in range(self._segments)
        ]

        # Delete the segment by converting it into a blank image following a top-down approach, i.e. in the list order
        # Since Numpy creates a view of the array when slicing, the original complete `im` array has been modified
        for i in range(self._to_delete):
            segmented[i][:] = 255

        # Reconstruct the image back to its Pillow representation
        return Image.fromarray(im)


class XORTransformation(MultiTransformation):
    """
    A XOR transformation finds the difference between two images to detect deletion of shapes between frames.
    """
    @property
    def name(self):
        return 'XOR'

    def apply(self, image, **kwargs):
        super(XORTransformation, self)._validate(**kwargs)

        # Convert each image into a black and white bi-level representation
        # Use a custom threshold since Pillow dithers the image adding noise to it
        # Reference: https://stackoverflow.com/a/50090612
        image = image.copy().point(lambda x: 255 if x > 200 else 0).convert('1')
        other = kwargs['other'].copy().point(lambda x: 255 if x > 200 else 0).convert('1')

        # Apply the logical XOR to find all pixels that are not present between the two frames,
        # convert it back to grayscale and apply a Gaussian blur to emphasize the differences
        # The blur is useful because some random pixels might be flagged as not being present
        # though these difference are not perceived by the human eye; but with the blur, only
        # the strongest differences are kept, e.g. a full white or black shape
        # The radius of the Gaussian was found arbitrarily via empirical experimentation
        return ImageChops.logical_xor(image, other).convert('L').filter(ImageFilter.GaussianBlur(radius=10))


class UnionTransformation(MultiTransformation):
    @property
    def name(self):
        return 'Union'

    def apply(self, image, **kwargs):
        super(UnionTransformation, self)._validate(**kwargs)

        # Convert each image into a Numpy array to perform the operation on the pixel values directly
        image_1 = np.array(image)
        other = np.array(kwargs['other'])

        # The union operation as defined by Kunda in his doctoral dissertation
        # Reference: https://smartech.gatech.edu/bitstream/handle/1853/47639/kunda_maithilee_201305_phd.pdf
        # Here we use minimum instead of maximum because Kunda assumed that the images had a value of 0 for white
        # but, in reality, 0 indicates a black pixel and 255 (or 1 if the image is binary) is white
        unified = np.minimum(image_1, other)

        # Reconstruct the image back to its Pillow representation
        return Image.fromarray(unified)
