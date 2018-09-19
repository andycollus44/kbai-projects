"""
This module contains classes to understand, operate and manipulate shapes inside Raven's figures.
"""
import math
import uuid
from collections import defaultdict, deque

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
CIRCLE = 'CIRCLE'

# Relative positions
INSIDE = 'INSIDE'
RIGHT_OF = 'RIGHT_OF'
LEFT_OF = 'LEFT_OF'
ABOVE = 'ABOVE'
BELOW = 'BELOW'


class RavensShape:
    """
    Represents a shape in a Raven's figure.
    """
    def __init__(self, contour):
        self._label = str(uuid.uuid4())
        self._contour = contour
        self._area_points = None
        self._bbox = self._bounding_box()
        self._moments = {}
        self._arc_length = self._perimeter()
        self._shape = None
        self._sides = 0
        self._positions = {}
        self._filled = False

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def contour(self):
        return self._contour

    @property
    def area_points(self):
        return self._area_points

    @area_points.setter
    def area_points(self, value):
        self._area_points = value

    @property
    def bbox(self):
        return self._bbox

    @property
    def moments(self):
        return self._moments

    @moments.setter
    def moments(self, value):
        self._moments = value

    @property
    def area(self):
        # The area of this shape computed via the raw moments
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Examples
        return self.moments['raw']['m00']

    @property
    def centroid(self):
        # The centroid of this shape computed via the raw moments
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Examples
        return (int(self.moments['raw']['m10'] / self.moments['raw']['m00']),
                int(self.moments['raw']['m01'] / self.moments['raw']['m00']))

    @property
    def angle(self):
        # Compute the "indicative" angle as proposed by Wobbrock et. al in their $1 Recognizer Algorithm
        # Reference: http://faculty.washington.edu/wobbrock/pubs/uist-07.01.pdf
        # The angle between the centroid of the shape and its first point of the contour
        return math.degrees(math.atan2(self.centroid[1] - self.contour[0][1], self.centroid[0] - self.contour[0][0]))

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
        return self.moments['hu']

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = value

    @property
    def filled(self):
        return self._filled

    @filled.setter
    def filled(self, value):
        self._filled = value

    def __repr__(self):
        return 'RavensShape(label={}, shape={}, filled={}, angle={}, positions={})'.format(
            self.label, self.shape, self.filled, self.angle, self.positions)

    def _bounding_box(self):
        # Computes the bounding box for a set of points
        minx, miny = np.min(self._contour, axis=0)
        maxx, maxy = np.max(self._contour, axis=0)

        return minx, miny, maxx, maxy

    def _perimeter(self):
        # The perimeter of the contour, or arc length, is simply the sum of distances between all pairs of points
        perimeter = 0
        prev = self._contour[0]

        for index in range(1, len(self._contour)):
            point = self._contour[index]
            dx = point[0] - prev[0]
            dy = point[1] - prev[1]

            perimeter = perimeter + math.sqrt(dx ** 2 + dy ** 2)

            prev = point

        return perimeter


class RavensShapeExtractor:
    # These thresholds were chosen arbitrarily after empirical experimentation
    _MINIMUM_NUMBER_OF_POINTS = 10
    _PERIMETER_THRESHOLD = 50
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
        binary_image = image.copy().point(lambda x: 255 if x > 60 else 0)
        # Also generate contours to be traced by the algorithm
        binary_contour = binary_image.filter(ImageFilter.CONTOUR)

        # Iteratively trace contours to extract shapes until no more contours are found
        shapes = []
        points = np.array(binary_contour).astype(np.int)

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

        # Compute the shapes' moments and the area points
        for shape in shapes:
            # Find all the points that lie inside the area described by the contour of the shape
            shape.area_points = _ContourAreaFinder.apply(shape)
            # Compute the shape's moments
            shape.moments = _Moments.compute(shape, image)

        # The contour tracing algorithm detects the inner and outer contours of an image when
        # it is not filled; however, these two detected shapes belong to the same one and should be deduped
        unique = self._dedupe_shapes(shapes)

        # Finally, compute some other attributes for all the shapes
        self._compute_attributes(unique, binary_image)

        return unique

    def _remove_small_shapes(self, shapes):
        return filter(lambda x: len(x) > self._MINIMUM_NUMBER_OF_POINTS, shapes)

    def _dedupe_shapes(self, shapes):
        # In order to dedupe, the centroid and the perimeter of the shapes can be compared relative to each other,
        # and the ones that are sufficiently close are deduped into a single shape

        unique = [shapes[0]]
        for index in range(1, len(shapes)):
            shape = shapes[index]
            shape_cx, shape_cy = shape.centroid

            # Check that the shape is not similar to any of the unique shapes identified so far
            is_duplicate = False

            for current in unique:
                current_cx, current_cy = current.centroid

                perimeter_diff = abs(shape.arclength - current.arclength)
                centroid_distance = math.sqrt((shape_cx - current_cx) ** 2 + (shape_cy - current_cy) ** 2)

                if perimeter_diff < self._PERIMETER_THRESHOLD and centroid_distance < self._CENTROID_DISTANCE_THRESHOLD:
                    # This is probably a duplicated shape
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            unique.append(shape)

        return unique

    def _compute_attributes(self, shapes, image):
        positions = _PositionFinder.apply(shapes)

        for shape in shapes:
            # Find the relative positions of each shape with respect to the others
            shape.positions = positions[shape.label]
            # Classify the unique shapes to find geometric relationships
            shape.shape, shape.sides = _ShapeClassifier.classify(shape.contour, shape.arclength)

        # Analyze "filled-ness" for the shapes which has to be done after all positions are computed
        # and the area points are obtained, see `_FilledAnalyzer.apply()` for an explanation of this
        for index, shape in enumerate(shapes):
            shape.filled = _FilledAnalyzer.apply(shape, shapes[index + 1:], image)


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
    _PERCENTAGE_OF_ORIGINAL_PERIMETER = 0.015
    # Number of points to resample for the contour chose arbitrarily after empirical experimentation
    _RESAMPLING = 128

    # Shapes with fixed number of vertices
    _SHAPES = {
        3: TRIANGLE,
        5: PENTAGON,
        6: HEXAGON,
        7: HEPTAGON,
        8: OCTAGON
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
        # Pre-process the contour by resampling the number of points to reduce noise
        resampled_contour = _ShapeClassifier._resample(contour, perimeter, _ShapeClassifier._RESAMPLING)
        # Now approximate the contour with the new resampled points, the new perimeter also needs to be computed
        approx_contour = _ShapeClassifier._approximate_contour(resampled_contour,
                                                               _ShapeClassifier._perimeter(resampled_contour))
        # Post-process the approximated points by merging points that are close to each other
        approx_contour = _ShapeClassifier._merge(approx_contour)
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

        if vertices > 8:
            # Assume it is a circle which does not have any sides
            return CIRCLE, 0

        return _ShapeClassifier._SHAPES[vertices], vertices

    @staticmethod
    def _resample(contour, perimeter, n):
        # Resamples the contour to have `n` equally distributed points
        # by following the proposed technique by Wobbrock et. al in their $1 Recognizer Algorithm
        # Reference: http://faculty.washington.edu/wobbrock/pubs/uist-07.01.pdf

        points = contour.tolist()
        threshold = perimeter / (n - 1)
        D = 0
        resampled = [(points[0][0], points[0][1])]

        index = 1

        while index < len(points):
            point = points[index]
            prev = points[index - 1]
            d = math.sqrt((prev[0] - point[0]) ** 2 + (prev[1] - point[1]) ** 2)

            if (D + d) >= threshold:
                qx = prev[0] + ((threshold - D) / d) * (point[0] - prev[0])
                qy = prev[1] + ((threshold - D) / d) * (point[1] - prev[1])

                resampled.append((qx, qy))
                # q will be the next point
                points.insert(index + 1, (qx, qy))

                D = 0
            else:
                D = D + d

            index = index + 1

        return np.array(resampled)

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
    def _merge(contour):
        # Keep a stack of the latest point
        merged = [contour[0]]

        for index in range(1, len(contour)):
            point = contour[index]
            current = merged[-1]
            distance = math.sqrt((point[0] - current[0]) ** 2 + (point[1] - current[1]) ** 2)

            if distance <= 5:
                # Merge these points together by obtaining the middle point
                middle_point = int((point[0] + current[0]) / 2), int((point[1] + current[1]) / 2)
                # Remove the current point and replace it with the middle point
                merged.pop()
                merged.append(middle_point)
                continue

            # This point does not need to be merged
            merged.append(point)

        return merged

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

    @staticmethod
    def _perimeter(contour):
        perimeter = 0
        prev = contour[0]

        for index in range(1, len(contour)):
            point = contour[index]
            dx = point[0] - prev[0]
            dy = point[1] - prev[1]

            perimeter = perimeter + math.sqrt(dx ** 2 + dy ** 2)

            prev = point

        return perimeter


class _PositionFinder:
    """
    Implements a position finder for a shape based on bounding boxes.
    """
    def __init__(self):
        pass

    @staticmethod
    def apply(shapes):
        """
        Finds all the positions for each shape with respect to the others.

        :param shapes: The shapes to find positions for.
        :type shapes: list[RavensShape]
        :return: A dictionary keyed by shape label with all positions.
        :rtype: dict
        """
        positions = {s.label: {} for s in shapes}

        for shape in shapes:
            shape_positions = defaultdict(list)

            for other in shapes:
                if shape.label == other.label:
                    continue

                if _PositionFinder._is_inside(other.bbox, shape.bbox):
                    shape_positions[INSIDE].append(other.label)
                elif _PositionFinder._is_right_of(other.bbox, shape.bbox):
                    shape_positions[RIGHT_OF].append(other.label)
                elif _PositionFinder._is_left_of(other.bbox, shape.bbox):
                    shape_positions[LEFT_OF].append(other.label)
                elif _PositionFinder._is_right_of(other.bbox, shape.bbox):
                    shape_positions[RIGHT_OF].append(other.label)
                elif _PositionFinder._is_above(other.bbox, shape.bbox):
                    shape_positions[ABOVE].append(other.label)
                elif _PositionFinder._is_below(other.bbox, shape.bbox):
                    shape_positions[BELOW].append(other.label)

            positions[shape.label] = dict(shape_positions.copy())

        return positions

    @staticmethod
    def _is_inside(bbox1, bbox2):
        # Determines whether the bounding box `bbox2` is inside `bbox1`
        minx_1, miny_1, maxx_1, maxy_1 = bbox1
        minx_2, miny_2, maxx_2, maxy_2 = bbox2

        return minx_2 >= minx_1 and maxx_2 <= maxx_1 and miny_2 >= miny_1 and maxy_2 <= maxy_1

    @staticmethod
    def _is_right_of(bbox1, bbox2):
        # Determines whether the bounding box `bbox2` is right of `bbox1`
        _, _, maxx_1, _ = bbox1
        minx_2, _, _, _ = bbox2

        return minx_2 >= maxx_1

    @staticmethod
    def _is_left_of(bbox1, bbox2):
        # Determines whether the bounding box `bbox2` is left of `bbox1`
        minx_1, _, _, _ = bbox1
        _, _, maxx_2, _ = bbox2

        return maxx_2 <= minx_1

    @staticmethod
    def _is_above(bbox1, bbox2):
        # Determines whether the bounding box `bbox2` is above `bbox1`
        _, miny_1, _, _ = bbox1
        _, _, _, maxy_2 = bbox2

        return maxy_2 <= miny_1

    @staticmethod
    def _is_below(bbox1, bbox2):
        # Determines whether the bounding box `bbox2` is below `bbox1`
        _, _, _, maxy_1 = bbox1
        _, miny_2, _, _ = bbox2

        return miny_2 >= maxy_1


class _FilledAnalyzer:
    """
    Implements an analyzer for the 'filled' attribute of a shape.
    """

    def __init__(self):
        pass

    @staticmethod
    def apply(shape, other_shapes, image):
        """
        Determines whether the given shape is filled or not in the original image.

        :param shape: The shape to determine its 'filled' attribute for.
        :type shape: RavensShape
        :param other_shapes: The other shapes from the same image.
        :type other_shapes: list[RavensShape]
        :param image: The original image, as a bi-level representation.
        :type image: PIL.Image.Image
        :return: True if the shape is filled, False otherwise.
        :rtype: bool
        """
        original = np.array(image.copy()).astype(np.int)

        # Step 1
        # Find any shapes that are inside the given shape
        inside_shapes = filter(lambda s: shape.label in s.positions.get(INSIDE, []), other_shapes)

        # Step 2
        # Make any inside shapes black (i.e. fill the shapes)
        for inside in inside_shapes:
            area = inside.area_points
            # If the shape does not have an area, skip it; this could happen for lines, for example
            if len(area) == 0:
                continue
            # In the image matrix, the row is the y-coordinate, while the column is the x-coordinate
            original[area[:, 1], area[:, 0]] = 0

        # Step 3
        # If all the points inside the given shape are black, then it means it was originally filled,
        # this comes the fact that if all the shapes inside are black, and the given shape is also black,
        # then the whole area will become black. If the shape is actually not filled, then there will be
        # a portion of the pixels that are white amongst all the other black ones. However, make sure
        # the shape actually has an area, it could happen that the area was not detected for this shape.
        # An example of this is the figures in Challenge Problem B-09 which is filled with lines instead
        # of a solid pattern!
        area = shape.area_points
        filled = len(area) > 0 and np.all(original[area[:, 1], area[:, 0]] == 0)

        return filled


class _ContourAreaFinder:
    """
    Implements a contour area finder based on a simple BFS.

    Reference: http://answers.opencv.org/question/45968/getting-area-points-inside-contour/ (see 3rd comment)
    """

    def __init__(self):
        pass

    @staticmethod
    def apply(shape):
        """
        Finds the area inside the contour, as a set of all points, for the given shape.

        :param shape: The shape to finds its contour area for.
        :type shape: RavensShape
        :return: All the points inside the contour area.
        :rtype: numpy.ndarray
        """
        # Convert the contour into a set of points for faster membership checks
        contour = set([(p[0], p[1]) for p in shape.contour])
        # Use an approximation of the centroid of the shape via its contour
        centroid = int(np.mean(shape.contour[:, 0])), int(np.mean(shape.contour[:, 1]))

        area = []

        # Start in the centroid of the shape
        queue = deque()
        queue.append(centroid)
        visited = set()

        while len(queue) > 0:
            point = queue.pop()

            # The point is part of the contour or has been visited, stop here
            if point in contour or point in visited:
                continue

            # Add it to the area
            area.append(point)
            visited.add(point)

            # Find all 4 neighbors (up, down, left, right) assuming (0,0) being the top left corner in the plane
            x, y = point
            for neighbor in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
                if neighbor not in visited:
                    queue.append(neighbor)

        return np.array(area)


class _Moments:
    """
    Implements a class for computing image moments.
    """

    def __init__(self):
        pass

    @staticmethod
    def compute(shape, image):
        """
        Computes the raw, central and Hu moments for the given shape.

        :param shape: The shape to compute moments for.
        :type shape: RavensShape
        :param image: The original image.
        :type image: PIL.Image.Image
        :return: The raw, central and Hu moments as a dictionary with keys: 'raw', 'central', 'hu'
                 where 'raw' and 'central' are dictionaries, and 'hu' is a list.
        :rtype: dict
        """
        image = np.array(image.copy()).astype(np.int)

        # Binarize the image by setting all points of the shape to 1 and everything else to 0
        image[:, :] = 0
        # Use the area of the shape, but if not available, then default to the contour,
        # this could happen when the shape extracted is a single line
        points = shape.area_points if len(shape.area_points) > 0 else shape.contour
        image[points[:, 1], points[:, 0]] = 1

        raw_moments = _Moments._compute_raw_moments(image, points)
        central_moments = _Moments._compute_central_moments(raw_moments)
        hu_moments = _Moments._compute_hu_moments(central_moments)

        return {
            'raw': raw_moments,
            'central': central_moments,
            'hu': hu_moments
        }

    @staticmethod
    def _compute_raw_moments(image, points):
        # Computes the raw moments of the given shape
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Raw_moments

        return {
            'm00': _Moments._compute_raw_moment(image, points, 0, 0),
            'm01': _Moments._compute_raw_moment(image, points, 0, 1),
            'm10': _Moments._compute_raw_moment(image, points, 1, 0),
            'm11': _Moments._compute_raw_moment(image, points, 1, 1),
            'm12': _Moments._compute_raw_moment(image, points, 1, 2),
            'm02': _Moments._compute_raw_moment(image, points, 0, 2),
            'm20': _Moments._compute_raw_moment(image, points, 2, 0),
            'm21': _Moments._compute_raw_moment(image, points, 2, 1),
            'm03': _Moments._compute_raw_moment(image, points, 0, 3),
            'm30': _Moments._compute_raw_moment(image, points, 3, 0)
        }

    @staticmethod
    def _compute_central_moments(raw_moments):
        # Computes the central moments of the shape
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Central_moments
        cx, cy = (int(raw_moments['m10'] / raw_moments['m00']),
                  int(raw_moments['m01'] / raw_moments['m00']))

        return {
            'mu00': raw_moments['m00'],
            'mu01': 0,
            'mu10': 0,
            'mu11': raw_moments['m11'] - cx * raw_moments['m01'],
            'mu20': raw_moments['m20'] - cx * raw_moments['m10'],
            'mu02': raw_moments['m02'] - cy * raw_moments['m01'],
            'mu21': (raw_moments['m21'] - 2 * cx * raw_moments['m11'] - cy * raw_moments['m20'] +
                     2 * cx * cx * raw_moments['m01']),
            'mu12': (raw_moments['m12'] - 2 * cy * raw_moments['m11'] - cx * raw_moments['m02'] +
                     2 * cy * cy * raw_moments['m10']),
            'mu30': raw_moments['m30'] - 3 * cx * raw_moments['m20'] + 2 * cx * cx * raw_moments['m10'],
            'mu03': raw_moments['m03'] - 3 * cy * raw_moments['m02'] + 2 * cy * cy * raw_moments['m01']
        }

    @staticmethod
    def _compute_hu_moments(central_moments):
        # Computes the Hu moments of the shape
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Moment_invariants
        n02 = _Moments._compute_invariant(central_moments, 0, 2)
        n03 = _Moments._compute_invariant(central_moments, 0, 3)
        n11 = _Moments._compute_invariant(central_moments, 1, 1)
        n12 = _Moments._compute_invariant(central_moments, 1, 2)
        n20 = _Moments._compute_invariant(central_moments, 2, 0)
        n21 = _Moments._compute_invariant(central_moments, 2, 1)
        n30 = _Moments._compute_invariant(central_moments, 3, 0)

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

    @staticmethod
    def _compute_raw_moment(image, points, p, q):
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Raw_moments
        x, y = points[:, 0], points[:, 1]

        return np.sum((x ** p) * (y ** q) * image[y, x])

    @staticmethod
    def _compute_invariant(central_moments, i, j):
        # Reference: https://en.wikipedia.org/wiki/Image_moment#Scale_invariants
        return central_moments['mu{}{}'.format(i, j)] / math.pow(central_moments['mu00'], (1 + (i + j) / 2))


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
