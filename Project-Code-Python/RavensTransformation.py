from abc import ABCMeta, abstractmethod, abstractproperty

from PIL import ImageOps


class Transformation:
    __metaclass__ = ABCMeta

    @abstractproperty
    def name(self):
        pass

    @abstractmethod
    def apply(self, image):
        """
        Applies this transformation to the given image.

        :param image: The image to apply this transformation to.
        :type image: PIL.Image.Image
        :return: The transformed image.
        :rtype: PIL.Image.Image
        """
        pass

    def __repr__(self):
        return self.name


class NoOpTransformation(Transformation):
    """
    A no-op transformation returns a copy of the given image.
    """
    @property
    def name(self):
        return 'NoOp'

    def apply(self, image):
        return image.copy()


class MirrorTransformation(Transformation):
    """
    A mirror transformation flips the image horizontally (left to right).
    """
    @property
    def name(self):
        return 'Mirror'

    def apply(self, image):
        return ImageOps.mirror(image)


class FlipTransformation(Transformation):
    """
    A flip transformation flips the image vertically (top to bottom).
    """
    @property
    def name(self):
        return 'Flip'

    def apply(self, image):
        return ImageOps.flip(image)
