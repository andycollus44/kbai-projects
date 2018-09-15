"""
This module contains classes to understand, operate and manipulate shapes inside Raven's figures.
"""

import math

import numpy as np
from PIL import ImageFilter

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
        self._moments = self._compute_moments()

    @property
    def points(self):
        return self._points

    @property
    def bbox(self):
        return self._bbox

    @property
    def area(self):
        # The area of this shape computed via the raw moments
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Examples
        return self._moments['m00']

    @property
    def centroid(self):
        # The centroid of this shape computed via the raw moments
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Examples
        return int(self._moments['m10'] / self._moments['m00']), int(self._moments['m01'] / self._moments['m00'])

    def _bounding_box(self):
        # Computes the bounding box for a set of points
        minx, miny = np.min(self._points, axis=0)
        maxx, maxy = np.max(self._points, axis=0)

        return minx, miny, maxx, maxy

    def _compute_moments(self):
        # Computes the raw moments of this shape
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Raw_moments
        return {
            'm00': self._compute_moment(0, 0),
            'm01': self._compute_moment(0, 1),
            'm10': self._compute_moment(1, 0)
        }

    def _compute_moment(self, p, q):
        # Pixel intensities are not needed to compute a raw moment because
        # the contour of a shape is found via black pixels which are zeroes
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Raw_moments
        return np.sum((self._points[:, 0] ** p) * (self._points[:, 1] ** q))


class RavensShapeExtractor:
    # These thresholds were chosen arbitrarily after empirical experimentation
    _MINIMUM_NUMBER_OF_POINTS = 10
    _AREA_THRESHOLD = 50
    _CENTROID_DISTANCE_THRESHOLD = 50

    def __init__(self):
        self._contour_tracer = _ContourTracer()

    def apply(self, image):
        """
        Extracts all shapes from the given image.

        :param image: The image to extract individual shapes from.
        :type image: PIL.Image.Image
        :return: A list of shapes
        :rtype: list[RavensShape]
        """

        # Convert each image into a black and white bi-level representation using a custom threshold since Pillow
        # dithers the image adding noise to it. A threshold of 60 was used because only the darkest contours should
        # be kept which helps separating shapes that are "joined" by some lighter grayish pixels that to the human eye
        # are white, but to the image processing library they are still darkish which affects the contour tracing
        # algorithm. So, anything that is not darker than 60 intensity value, i.e. very black, is considered white.
        # 60 was chosen arbitrarily after empirical experimentation
        # Reference: https://stackoverflow.com/a/50090612
        image = image.copy().point(lambda x: 255 if x > 60 else 0).convert('1')
        # Also generate contours to be traced by the algorithm
        image = image.filter(ImageFilter.CONTOUR)

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

        # No shapes present in the image!
        if len(shapes) == 0:
            return []

        # Post-process shapes by removing any "suspicious" shapes with very few amounts of points
        shapes = self._remove_small_shapes(shapes)
        shapes = [RavensShape(shape) for shape in shapes]

        # Finally, the contour tracing algorithm detects the inner and outer contours of an image when
        # it is not filled; however, these two detected shapes belong to the same one and should be deduped
        unique = self._dedupe_shapes(shapes)

        return unique

    def _remove_small_shapes(self, shapes):
        return filter(lambda x: len(x) > self._MINIMUM_NUMBER_OF_POINTS, shapes)

    def _dedupe_shapes(self, shapes):
        # In order to dedupe, the centroid and the area of the shapes can be compared relative to each other,
        # and the ones that are sufficiently close are deduped into a single shape

        # Keep a stack of the unique shapes
        unique = [shapes[0]]
        for index in range(1, len(shapes)):
            shape = shapes[index]
            shape_cx, shape_cy = shape.centroid

            current = unique[-1]
            current_cx, current_cy = current.centroid

            area_difference = abs(shape.area - current.area)
            centroid_distance = math.sqrt((shape_cx - current_cx) ** 2 + (shape_cy - current_cy) ** 2)

            if area_difference < self._AREA_THRESHOLD and centroid_distance < self._CENTROID_DISTANCE_THRESHOLD:
                # This is probably a duplicated shape
                continue

            unique.append(shape)

        return unique


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
