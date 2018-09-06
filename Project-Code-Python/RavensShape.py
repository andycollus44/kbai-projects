"""
This module contains classes to understand, operate and manipulate shapes inside Raven's figures.
"""

import numpy as np
from PIL import Image, ImageFilter

from RavensFigure import RavensFigure

# Absolute directions
_N = 0
_NE = 1
_E = 2
_SE = 3
_S = 4
_SW = 5
_W = 6
_NW = 7

# Relative directions
_F = 0   # Front
_FR = 1  # Front-Right
_R = 2   # Right
_RB = 3  # Right-Back
_B = 4   # Back
_LB = 5  # Left-Back
_L = 6   # Left
_FL = 7  # Front-Left

# Pixel intensities
_BLACK = 0


class RavensShape:
    """
    Represents a shape in a Raven's figure.
    """
    def __init__(self, points):
        self._points = points
        self._bbox = self._bounding_box()

    @property
    def points(self):
        return self._points

    @property
    def bbox(self):
        return self._bbox

    def _bounding_box(self):
        # Computes the bounding box for a set of points
        minx, miny = np.min(self._points, axis=0)
        maxx, maxy = np.max(self._points, axis=0)

        return minx, miny, maxx, maxy


class RavensShapeExtractor:
    _MINIMUM_NUMBER_OF_POINTS = 10

    def __init__(self):
        self._contour_tracer = _ContourTracer()

    def apply(self, figure):
        """
        Extracts all shapes from the given figure.

        :param figure: The figure to extract individual shapes from.
        :type figure: RavensFigure
        :return: A list of shapes
        :rtype: list[RavensShape]
        """
        assert isinstance(figure, RavensFigure)

        # Perform the following transformations to the image:
        # 1. Convert to grayscale ('L')
        # 2. Reduce size of the image to 32 x 32 so that operations are faster
        # 3. Generate contours of shapes
        # 4. Convert to bi-level (1's and 0's) for shape extraction
        image = (Image.open(figure.visualFilename)
                      .convert('L').resize((32, 32), resample=Image.BICUBIC)
                      .filter(ImageFilter.CONTOUR)
                      .convert('1'))

        # Iteratively trace contours to extract shapes until no more contours are found
        shapes = []
        points = np.array(image).astype(np.int)

        while True:
            contour = self._contour_tracer.apply(points)

            if len(contour) == 0:
                break

            # Whiten all the points for the extracted shape to "remove" them from the next iteration
            points[contour[:, 1], contour[:, 0]] = 1
            shapes.append(contour)

        # Post-process shapes by removing any "suspicious" shapes with very few amounts of points
        shapes = filter(lambda x: len(x) > self._MINIMUM_NUMBER_OF_POINTS, shapes)

        return [RavensShape(shape) for shape in shapes]


class _ContourTracer:
    """
    Implements the fast contour-tracing algorithm proposed by Jonghoon, Seo et. al.

    Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4813928/ (see Algorithm 3).
    """
    def __init__(self):
        pass

    def apply(self, points):
        # Initialize the tracer by finding the first point that is 'black'
        black_points = np.argwhere(points == _BLACK)

        if len(black_points) == 0:
            # No more contours present!
            return []

        init = black_points[0]

        # In the image matrix, the row is the y-coordinate, while the column is the x-coordinate
        row, col = init
        tracer = _Tracer(_Point(col, row))

        while True:
            # Stage 1
            point = tracer.point
            direction = tracer.direction

            if self._is_black(points, point.left_back(direction)):
                if self._is_black(points, point.left(direction)):
                    # Case 1
                    tracer + point.left(direction)
                    tracer.move(_L)

                    tracer + tracer.point.left(tracer.direction)
                    tracer.move(_L)
                else:
                    # Case 2
                    tracer + point.left_back(direction)
                    tracer.move(_B)
            else:
                if self._is_black(points, point.left(direction)):
                    # Case 3
                    tracer + point.left(direction)
                    tracer.move(_L)

            # Stage 2
            point = tracer.point
            direction = tracer.direction

            if self._is_black(points, point.front_left(direction)):
                if self._is_black(points, point.front(direction)):
                    # Case 6
                    tracer + point.front(direction)
                    tracer.move(_L)

                    tracer + tracer.point.front(tracer.direction)
                    tracer.move(_R)
                else:
                    # Case 5
                    tracer + point.front_left(direction)
            elif self._is_black(points, point.front(direction)):
                # Case 7
                tracer + point.front(direction)
                tracer.move(_R)
            else:
                # Case 8
                tracer.move(_B)

            if tracer.point == tracer.initial_point and tracer.direction == tracer.initial_direction:
                # The tracer has returned to the initial point, so the contour has been found
                break

        # Return the contour as a numpy array of xy tuples
        return np.array([p.as_tuple() for p in tracer.points])

    def _is_black(self, points, point):
        # In the image matrix the x-coordinate is the column, while the y-coordinate is the row
        x, y = point[0], point[1]

        try:
            return points[y, x] == _BLACK
        except IndexError:
            # The point is outside the image, skip it
            return False


class _Tracer:
    """
    Represents a contour tracer.

    A tracer is always initialized facing EAST.
    """

    # Defines how the tracer moves given a relative direction and its current absolute direction
    _MOVES = {
        _L: {
            _N: _W,
            _E: _N,
            _S: _E,
            _W: _S
        },
        _R: {
            _N: _E,
            _E: _S,
            _S: _W,
            _W: _N
        },
        _B: {
            _N: _S,
            _E: _W,
            _S: _N,
            _W: _E
        }
    }

    def __init__(self, init):
        self._points = [init]
        self._direction = _E
        self._initial_direction = self._direction

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, new_direction):
        self._direction = new_direction

    @property
    def initial_direction(self):
        return self._initial_direction

    @property
    def points(self):
        return self._points

    @property
    def point(self):
        """
        Returns the most recent point.

        :return: A point
        :rtype: _Point
        """
        return self._points[-1]

    @property
    def initial_point(self):
        """
        Returns the first point.

        :return: A point
        :rtype: _Point
        """
        return self._points[0]

    def __add__(self, point):
        """
        Adds a point to the list of traced points.

        :param point: The point to add.
        """
        self._points.append(point)

    def move(self, relative_direction):
        """
        Moves the tracer's absolute direction given the relative direction.

        :param relative_direction: The relative direction to move the tracer to.
        """
        self._direction = self._MOVES[relative_direction][self._direction]


class _Point:
    """
    Represents a 2D point coordinate with methods to discover neighbors in different directions.
    """

    # Defines relative changes of point coordinates based on absolute positions
    _DELTAS = {
        _N: {
            _F: (0, -1),
            _FR: (1, -1),
            _R: (1, 0),
            _RB: (1, 1),
            _B: (0, 1),
            _LB: (-1, 1),
            _L: (-1, 0),
            _FL: (-1, -1)
        },
        _E: {
            _F: (1, 0),
            _FR: (1, 1),
            _R: (0, 1),
            _RB: (-1, 1),
            _B: (-1, 0),
            _LB: (-1, -1),
            _L: (0, -1),
            _FL: (1, -1)
        },
        _S: {
            _F: (0, 1),
            _FR: (-1, 1),
            _R: (-1, 0),
            _RB: (-1, -1),
            _B: (0, -1),
            _LB: (1, -1),
            _L: (1, 0),
            _FL: (1, 1)
        },
        _W: {
            _F: (-1, 0),
            _FR: (-1, -1),
            _R: (0, -1),
            _RB: (1, -1),
            _B: (1, 0),
            _LB: (1, 1),
            _L: (0, 1),
            _FL: (-1, 1)
        }
    }

    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __eq__(self, other):
        return self._x == other.x and self._y == other.y

    def __getitem__(self, item):
        return self._x if item == 0 else self._y

    def __repr__(self):
        return 'Point({}, {})'.format(self._x, self._y)

    def front(self, absolute_direction):
        """
        Returns the point in front of this point based on the given absolute direction.

        :param absolute_direction: The current absolute direction.
        :return: A new point
        """
        return self._apply_delta(absolute_direction, _F)

    def front_left(self, absolute_direction):
        """
        Returns the point in front and to the left of this point based on the given absolute direction.

        :param absolute_direction: The current absolute direction.
        :return: A new point
        """
        return self._apply_delta(absolute_direction, _FL)

    def left(self, absolute_direction):
        """
        Returns the point to the left of this point based on the given absolute direction.

        :param absolute_direction: The current absolute direction.
        :return: A new point
        """
        return self._apply_delta(absolute_direction, _L)

    def left_back(self, absolute_direction):
        """
        Returns the point to the left and back of this point based on the given absolute direction.

        :param absolute_direction: The current absolute direction.
        :return: A new point
        """
        return self._apply_delta(absolute_direction, _LB)

    def back(self, absolute_direction):
        """
        Returns the point back of this point based on the given absolute direction.

        :param absolute_direction: The current absolute direction.
        :return: A new point
        """
        return self._apply_delta(absolute_direction, _B)

    def as_tuple(self):
        """
        Converts this point to a tuple representing its coordinate.

        :return: A tuple.
        """
        return self._x, self._y

    def _apply_delta(self, absolute_direction, relative_direction):
        dx, dy = self._DELTAS[absolute_direction][relative_direction]
        return _Point(self._x + dx, self._y + dy)
