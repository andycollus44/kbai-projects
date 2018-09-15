from abc import ABCMeta, abstractmethod, abstractproperty

from PIL import ImageChops, ImageFilter, ImageOps, Image, ImageDraw

from RavensShape import RavensShapeExtractor

# Types of transformations with respect to the images they act on
SINGLE = 'SINGLE'
MULTI = 'MULTI'


class Transformation:
    __metaclass__ = ABCMeta

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def type(self):
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


class MultiTransformation(Transformation):
    __metaclass__ = ABCMeta

    @property
    def type(self):
        return MULTI

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
            draw.polygon(shape.points.flatten().tolist(), fill=0, outline=255)

        return reconstructed


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
