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

# Shapes
BLOB = 'BLOB'
TRIANGLE = 'TRIANGLE'
SQUARE = 'SQUARE'
RECTANGLE = 'RECTANGLE'
PENTAGON = 'PENTAGON'
HEXAGON = 'HEXAGON'
HEPTAGON = 'HEPTAGON'
OCTAGON = 'OCTAGON'
NONAGON = 'NONAGON'
DECAGON = 'DECAGON'
CIRCLE = 'CIRCLE'


class RavensShape:
    """
    Represents a shape in a Raven's figure.
    """
    def __init__(self, points):
        self._points = points
        self._bbox = self._bounding_box()
        self._raw_moments = self._compute_raw_moments()
        self._central_moments = self._compute_central_moments()
        self._hu_moments = self._compute_hu_moments()
        self._arc_length = self._perimeter()
        self._shape = None
        self._sides = 0

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
        return self._raw_moments['m00']

    @property
    def centroid(self):
        # The centroid of this shape computed via the raw moments
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Examples
        return (int(self._raw_moments['m10'] / self._raw_moments['m00']),
                int(self._raw_moments['m01'] / self._raw_moments['m00']))

    @property
    def arclength(self):
        return self._arc_length

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def sides(self):
        return self._sides

    @sides.setter
    def sides(self, value):
        self._sides = value

    @property
    def hu_moments(self):
        return self._hu_moments

    def _bounding_box(self):
        # Computes the bounding box for a set of points
        minx, miny = np.min(self._points, axis=0)
        maxx, maxy = np.max(self._points, axis=0)

        return minx, miny, maxx, maxy

    def _perimeter(self):
        # The perimeter of the contour, or arc length, is simply the sum of distances between all pairs of points
        perimeter = 0
        prev = self._points[0]

        for index in range(1, len(self._points)):
            point = self._points[index]
            dx = point[0] - prev[0]
            dy = point[1] - prev[1]

            perimeter = perimeter + math.sqrt(dx ** 2 + dy ** 2)

            prev = point

        return perimeter

    def _compute_raw_moments(self):
        # Computes the raw moments of this shape
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Raw_moments
        return {
            'm00': self._compute_raw_moment(0, 0),
            'm01': self._compute_raw_moment(0, 1),
            'm10': self._compute_raw_moment(1, 0),
            'm11': self._compute_raw_moment(1, 1),
            'm12': self._compute_raw_moment(1, 2),
            'm02': self._compute_raw_moment(0, 2),
            'm20': self._compute_raw_moment(2, 0),
            'm21': self._compute_raw_moment(2, 1),
            'm03': self._compute_raw_moment(0, 3),
            'm30': self._compute_raw_moment(3, 0)
        }

    def _compute_central_moments(self):
        # Computes the central moments of this shape
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Central_moments
        cx, cy = self.centroid

        return {
            'mu00': self._raw_moments['m00'],
            'mu01': 0,
            'mu10': 0,
            'mu11': self._raw_moments['m11'] - cx * self._raw_moments['m01'],
            'mu20': self._raw_moments['m20'] - cx * self._raw_moments['m10'],
            'mu02': self._raw_moments['m02'] - cy * self._raw_moments['m01'],
            'mu21': (self._raw_moments['m21'] - 2 * cx * self._raw_moments['m11'] - cy * self._raw_moments['m20'] +
                     2 * cx * cx * self._raw_moments['m01']),
            'mu12': (self._raw_moments['m12'] - 2 * cy * self._raw_moments['m11'] - cx * self._raw_moments['m02'] +
                     2 * cy * cy * self._raw_moments['m10']),
            'mu30': (self._raw_moments['m30'] - 3 * cx * self._raw_moments['m20'] +
                     2 * cx * cx * self._raw_moments['m10']),
            'mu03': (self._raw_moments['m03'] - 3 * cy * self._raw_moments['m02'] +
                     2 * cy * cy * self._raw_moments['m01'])
        }

    def _compute_hu_moments(self):
        # Computes the Hu moments of this shape
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Moment_invariants
        n02 = self._compute_invariant(0, 2)
        n03 = self._compute_invariant(0, 3)
        n11 = self._compute_invariant(1, 1)
        n12 = self._compute_invariant(1, 2)
        n20 = self._compute_invariant(2, 0)
        n21 = self._compute_invariant(2, 1)
        n30 = self._compute_invariant(3, 0)

        return [
            n20 + n02,
            (n20 - n02) ** 2 + 4 * n11 * n11,
            (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
            (n30 + n12) ** 2 + (n21 + n03) ** 2,
            ((n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) +
             (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)),
            (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03),
            ((3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) -
             (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
        ]

    def _compute_raw_moment(self, p, q):
        # Pixel intensities are not needed to compute a raw moment because
        # the contour of a shape is found via black pixels which are zeroes
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Raw_moments
        return np.sum((self._points[:, 0] ** p) * (self._points[:, 1] ** q))

    def _compute_invariant(self, i, j):
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Scale_invariants
        return self._central_moments['mu{}{}'.format(i, j)] / math.pow(self._central_moments['mu00'], (1 + (i + j) / 2))


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
        image = image.copy().point(lambda x: 255 if x > 60 else 0)
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

        # Now classify the unique shapes to find geometric relationships
        for shape in unique:
            shape.shape, shape.sides = _ShapeClassifier.classify(shape.points, shape.arclength)

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


class RavensShapeMatcher:
    """
    Implements a shape matcher based on Hu moments.

    Reference: https://tinyurl.com/y84hmhdq (see CV_CONTOURS_MATCH_I1)
    """

    def __init__(self):
        pass

    def apply(self, shape, other_shape):
        """
        Matches the given two shapes returning a similarity measure.

        :param shape: One of the shapes to match.
        :type shape: RavensShape
        :param other_shape: The other shape to match.
        :type other_shape: RavensShape
        :return: A similarity measure where 1.0 is a perfect match and 0.0 is no match at all.
        :rtype: float
        """
        hu_a = np.array(shape.hu_moments)
        hu_b = np.array(other_shape.hu_moments)

        m_a = np.sign(hu_a) * np.log10(np.fabs(hu_a))
        m_b = np.sign(hu_b) * np.log10(np.fabs(hu_b))

        return max(0.0, 1.0 - np.sum(np.fabs((1. / m_a) - (1. / m_b))))


class _ShapeClassifier:
    """
    Implements a shape classifier based on contour approximation.
    """
    # Percentage of the original perimeter to keep chosen arbitrarily after empirical experimentation
    _PERCENTAGE_OF_ORIGINAL_PERIMETER = 0.009

    # Shapes with fixed number of vertices
    _SHAPES = {
        3: TRIANGLE,
        5: PENTAGON,
        6: HEXAGON,
        7: HEPTAGON,
        8: OCTAGON,
        9: NONAGON,
        10: DECAGON
    }

    def __init__(self):
        pass

    @staticmethod
    def classify(contour, perimeter):
        """
        Classifies a shape given a set of points (contour).

        :param contour: The set of points describing the contour of the shape.
        :type contour: list
        :param perimeter: The arc length of the contour used for approximation.
        :type perimeter: float
        :return: A tuple (shape, sides)
        :rtype: tuple
        """
        approx_contour = _ShapeClassifier._approximate_contour(contour, perimeter)
        # Remove the last point since the contour is closed, and thus the last point is repeated
        approx_contour = [(point[0], point[1]) for point in approx_contour[:-1]]

        vertices = len(approx_contour)

        # Classify the shape based on number of vertices, this assumes shapes are only geometric!
        if vertices < 3:
            # Unidentified object
            return BLOB, -1

        if vertices == 4:
            # It is either a square or a rectangle, the aspect ratio of a square should be closer to one
            aspect_ratio = _ShapeClassifier._aspect_ratio(approx_contour)
            return (SQUARE, vertices) if 0.95 <= aspect_ratio <= 1.05 else (RECTANGLE, vertices)

        if vertices > 10:
            # Assume it is a circle which does not have any sides
            return CIRCLE, 0

        return _ShapeClassifier._SHAPES[vertices], vertices

    @staticmethod
    def _approximate_contour(contour, perimeter):
        # Approximates the contour by implementing the Ramer-Douglas-Peucker algorithm
        # Modified to be a vectorized version to leverage the power of Numpy
        # Reference: https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm

        # Find the point with the maximum distance to the line described by the first and the last points
        distances = _ShapeClassifier._distance_to_line(contour, contour[0], contour[-1])
        index = np.argmax(distances)
        max_distance = distances[index]

        # If the max distance is greater than epsilon, then recursively simplify the contour
        if max_distance > _ShapeClassifier._PERCENTAGE_OF_ORIGINAL_PERIMETER * perimeter:
            result_1 = _ShapeClassifier._approximate_contour(contour[:index + 1], perimeter)
            result_2 = _ShapeClassifier._approximate_contour(contour[index:], perimeter)

            # Merge
            result = result_1[:-1] + result_2
        else:
            # Keep these points
            result = [contour[0], contour[-1]]

        return result

    @staticmethod
    def _distance_to_line(points, line1, line2):
        # The distances from the points to a line defined by points `line1` and `line2`
        # Reference: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points

        if np.all(line1 == line2):
            # Handle closed contour cases where the points of the line are the same
            # The norm of the vectors can be used as the distance
            # Reference: https://tinyurl.com/y78klmch
            return np.linalg.norm(points - line1, axis=1)

        return np.divide(abs(np.cross(line2 - line1, line1 - points)), np.linalg.norm(line2 - line1))

    @staticmethod
    def _aspect_ratio(contour):
        minx, miny = np.min(contour, axis=0)
        maxx, maxy = np.max(contour, axis=0)
        width = maxx - minx
        height = maxy - miny

        return float(width) / float(height)


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
