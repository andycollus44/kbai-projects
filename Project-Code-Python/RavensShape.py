"""
This module contains classes to understand, operate and manipulate shapes inside Raven's figures.
"""
import json
import math
import uuid
from collections import defaultdict, deque, namedtuple
from operator import attrgetter

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
        self._size_rank = 0

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
    def width(self):
        minx, _, maxx, _ = self._bbox
        return maxx - minx

    @property
    def height(self):
        _, miny, _, maxy = self._bbox
        return maxy - miny

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

    @property
    def size_rank(self):
        return self._size_rank

    @size_rank.setter
    def size_rank(self, value):
        self._size_rank = value

    def __repr__(self):
        return 'RavensShape(label={}, shape={}, filled={}, angle={}, size_rank={}, positions={})'.format(
            self.label, self.shape, self.filled, self.angle, self.size_rank, self.positions)

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
    _CENTROID_DISTANCE_THRESHOLD = 50

    def __init__(self):
        self._contour_tracer = _ContourTracer()
        self._shape_classifier = RavensShapeTemplateClassifier()

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

                perimeters_are_similar = _perimeters_are_similar(shape.arclength, current.arclength)
                centroid_distance = math.sqrt((shape_cx - current_cx) ** 2 + (shape_cy - current_cy) ** 2)

                if perimeters_are_similar and centroid_distance < self._CENTROID_DISTANCE_THRESHOLD:
                    # This is probably a duplicated shape
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            unique.append(shape)

        return unique

    def _compute_attributes(self, shapes, image):
        positions = _PositionFinder.apply(shapes)
        ranks = RavensRelativeSizeRanker.apply(shapes)

        for shape in shapes:
            # Find the relative positions of each shape with respect to the others
            shape.positions = positions[shape.label]
            # Find the relative rank for the size of each shape
            shape.size_rank = ranks[shape.label]
            # Classify the unique shapes to find geometric relationships
            shape.shape, shape.sides, _ = self._shape_classifier.classify(shape)

        # Analyze "filled-ness" for the shapes which has to be done after all positions are computed
        # and the area points are obtained, see `_FilledAnalyzer.apply()` for an explanation of this
        for index, shape in enumerate(shapes):
            shape.filled = _FilledAnalyzer.apply(shape, shapes[index + 1:], image)


class RavensShapeMatcher:
    """
    Implements a shape matcher based on different attributes.
    """
    MATCH_ALL = 0

    def __init__(self):
        pass

    def apply(self, shape, other_shape, match=MATCH_ALL):
        """
        Matches the given two shapes returning a similarity measure.

        :param shape: One of the shapes to match.
        :type shape: RavensShape
        :param other_shape: The other shape to match.
        :type other_shape: RavensShape
        :param match: The type of match to perform. Valid values are: MATCH_ALL.
        :type match: int
        :return: A similarity measure where 1.0 is a perfect match and 0.0 is no match at all.
        :rtype: float
        """
        if match == self.MATCH_ALL:
            # Compare the shape, the filled-ness and the perimeter
            attributes = [
                shape.shape == other_shape.shape,
                shape.filled == other_shape.filled,
                _perimeters_are_similar(shape.arclength, other_shape.arclength)
            ]
        else:
            raise ValueError('Invalid match: {}'.format(match))

        return self._similarity(attributes)

    def _similarity(self, expected):
        return sum(expected) / float(len(expected))


class RavensRelativeSizeRanker:
    """
    Implements a ranker to assign relative ranks to shapes based on their perimeters.
    """

    # This threshold was chose arbitrarily after empirical experimentation
    _PERIMETER_THRESHOLD = 50

    def __init__(self):
        pass

    @staticmethod
    def apply(shapes):
        """
        Ranks each shape based on its size, relative to the other shapes where lower ranks mean smaller shapes.

        :param shapes: The shapes to rank.
        :type shapes: list[RavensShape]
        :return: A dictionary of sizes keyed by each shape's label.
        :rtype: dict
        """
        # Sort all shapes in increasing order of their perimeter
        sorted_shapes = sorted(shapes, key=attrgetter('arclength'))

        # Assign ranks to reach shape relative to each other
        rank = 0
        # This stack will hold the latest shape
        assigned = [sorted_shapes[0]]
        sizes = {assigned[0].label: rank}

        for shape in sorted_shapes[1:]:
            latest = assigned[-1]
            diff = abs(shape.arclength - latest.arclength)

            if diff < RavensRelativeSizeRanker._PERIMETER_THRESHOLD:
                # The current shape is similar in size to the latest one, so assign the same rank
                sizes[shape.label] = latest.size_rank
                continue

            # The current shape is significantly different in size, i.e. larger, so increment the rank
            # assign that rank to the label, and now that shape becomes the latest one for next iteration
            rank = rank + 1
            sizes[shape.label] = rank
            assigned.append(shape)

        return sizes


class RavensShapeTemplateClassifier:
    """
    Implements a classifier that uses templates to assign shapes to contours.

    This is an implementation of the $1 Recognizer Algorithm as proposed by Wobbrock et. al.
    Reference: http://faculty.washington.edu/wobbrock/pubs/uist-07.01.pdf
    """
    _TEMPLATES_FILE = 'shapes.txt'
    _SIDES = {
        'TRIANGLE': 3,
        'SQUARE': 4,
        'PENTAGON': 5,
        'HEXAGON': 6,
        'HEPTAGON': 7,
        'OCTAGON': 8
    }
    _RESAMPLING = 64
    _SCALE = 250.0

    Template = namedtuple('Template', ['name', 'points'])

    def __init__(self):
        # Load all existing templates, if any
        try:
            with open(self._TEMPLATES_FILE) as f:
                templates = json.load(f)
                self._templates = [self.Template(name=t['name'], points=t['points']) for t in templates]
        except IOError:
            # The file with templates does not exist
            self._templates = []

    @property
    def templates(self):
        return self._templates

    def save_templates(self, templates):
        """
        Saves the given templates to the file.

        :param templates: The templates to save.
        :type templates: list[Template]
        """
        for template in templates:
            points = self._preprocess([(p[0], p[1]) for p in template.points])
            self._templates.append(self.Template(name=template.name, points=points))

        # Save them to our "database" of known templates of shapes
        with open(self._TEMPLATES_FILE, 'w') as out:
            json.dump([dict(t._asdict()) for t in self._templates], out)

    def classify(self, shape):
        """
        Classifies the given shape's contour into an actual shape, e.g. a circle.

        :param shape: The input shape whose contour will be classified.
        :type shape: RavensShape
        :return: The name of the shape, its number of sides (0 for non-geometric figureS) and its similarity score.
        #:rtype: tuple
        """
        # Handle lines gracefully
        if self._is_line([(p[0], p[1]) for p in shape.contour]):
            return 'LINE', 0, 1.0

        contour = self._preprocess([(p[0], p[1]) for p in shape.contour])

        min_distance = float('inf')
        chosen_template = None

        for template in self._templates:
            # For each template perform a "Golden Ratio" search to find the distance from the shape's contour to it
            distance = self._distance_at_best_angle(contour, template.points, -math.radians(45.0), math.radians(45.0),
                                                    math.radians(2.0))

            if distance < min_distance:
                # The closer the distance is, the better the shape is matching against the template
                min_distance = distance
                chosen_template = template.name

        score = 1. - min_distance / (0.5 * math.sqrt(self._SCALE ** 2 + self._SCALE ** 2))

        # Return the name of the template, its number of sides if it a geometric figure and its similarity score
        return chosen_template, self._SIDES.get(chosen_template, 0), score

    def _is_line(self, points):
        # If the width or the height is 0, it means it is a line
        minx, miny, maxx, maxy = self._bounding_box(points)
        width = maxx - minx
        height = maxy - miny

        return width == 0 or height == 0

    def _preprocess(self, points):
        # Applies all pre-processing steps to the given set of points
        points = self._resample(points, self._RESAMPLING)
        points = self._rotate_to_zero(points)
        points = self._scale_to(points, self._SCALE)
        points = self._translate_to_origin(points)

        return points

    def _resample(self, points, n):
        # Resamples the given set of points to have `n` roughly equally separated points
        interval_length = self._path_length(points) / (n - 1)
        distance = 0.
        resampled = [(points[0][0], points[0][1])]

        i = 1

        while i < len(points):
            prev = points[i - 1]
            point = points[i]

            dist = self._distance(prev, point)

            if (distance + dist) >= interval_length:
                qx = prev[0] + ((interval_length - distance) / dist) * (point[0] - prev[0])
                qy = prev[1] + ((interval_length - distance) / dist) * (point[1] - prev[1])

                resampled.append((qx, qy))
                # qx,qy will act as the new previous point in the next iteration
                points.insert(i, (qx, qy))
                distance = 0.
            else:
                distance = distance + dist

            i = i + 1

        # Sometimes we fall a rounding-error short of adding the last point, so add it if so
        if len(resampled) == n - 1:
            resampled.append((points[-1][0], points[-1][1]))

        return resampled

    def _rotate_to_zero(self, points):
        # Rotates the given set of points around their centroid to make the shape's indicative angle zero
        centroid = self._centroid(points)
        relative_angle = math.atan2(centroid[1] - points[0][1], centroid[0] - points[0][0])

        return self._rotate_by(points, -relative_angle)

    def _rotate_by(self, points, radians):
        # Rotates the given set of points around their centroid `radians` amount
        centroid = self._centroid(points)
        cos = math.cos(radians)
        sin = math.sin(radians)

        return [
            ((p[0] - centroid[0]) * cos - (p[1] - centroid[1]) * sin + centroid[0],
             (p[0] - centroid[0]) * sin + (p[1] - centroid[1]) * cos + centroid[1])
            for p in points
        ]

    def _scale_to(self, points, size):
        # Scales the given set of points so that their bounding box is of size `size ^ 2`
        minx, miny, maxx, maxy = self._bounding_box(points)
        width = maxx - minx
        height = maxy - miny

        return [(p[0] * (size / float(width)), p[1] * (size / float(height))) for p in points]

    def _translate_to_origin(self, points):
        # Translates the given set of points back to the origin
        centroid = self._centroid(points)
        return [(p[0] - centroid[0], p[1] - centroid[1]) for p in points]

    def _distance_at_best_angle(self, points, template_points, radians_a, radians_b, radians_delta):
        # Performs a "Golden Ratio" search over a set of angles to find the minimum distance between
        # the given points and the template, i.e. find the template that best matches the given shape
        phi = 0.5 * (-1. + math.sqrt(5.))

        x1 = phi * radians_a + (1. - phi) * radians_b
        f1 = self._distance_at_angle(points, template_points, x1)

        x2 = (1. - phi) * radians_a + phi * radians_b
        f2 = self._distance_at_angle(points, template_points, x2)

        while abs(radians_b - radians_a) > radians_delta:
            if f1 < f2:
                radians_b = x2
                x2 = x1
                f2 = f1
                x1 = phi * radians_a + (1. - phi) * radians_b
                f1 = self._distance_at_angle(points, template_points, x1)
            else:
                radians_a = x1
                x1 = x2
                f1 = f2
                x2 = (1. - phi) * radians_a + phi * radians_b
                f2 = self._distance_at_angle(points, template_points, x2)

        return min(f1, f2)

    def _distance_at_angle(self, points, other_points, radians):
        # Rotates the given set of points by `radians` amount,
        # and then computes the distance between said points and others
        rotated = self._rotate_by(points, radians)
        return self._path_distance(rotated, other_points)

    def _bounding_box(self, points):
        points = np.array(points)
        minx, miny = np.min(points, axis=0)
        maxx, maxy = np.max(points, axis=0)

        return minx, miny, maxx, maxy

    def _centroid(self, points):
        return tuple(np.mean(np.array(points), axis=0))

    def _path_length(self, points):
        return sum([self._distance(points[i - 1], points[i]) for i in range(1, len(points))])

    def _path_distance(self, points_a, points_b):
        assert len(points_a) == len(points_b), 'Points do not have the same length {} != {}'.format(len(points_a),
                                                                                                    len(points_b))
        distance = sum([self._distance(points_a[i], points_b[i]) for i in range(0, len(points_a))])

        return distance / len(points_a)

    def _distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


class RavensShapeIdentifier:
    """
    Identifies shapes inside the given image via its bounding boxes.

    Note: This class does not perform any actual extraction of the shapes or computation of any other attributes.
          If a more sophisticated extraction is needed then use the `RavensShapeExtractor`.
    """

    def __init__(self):
        self._connected_components = _ConnectedComponentsAlgorithm(_ConnectedComponentsAlgorithm.CONNECTIVITY_FOUR)

    def apply(self, image):
        """
        Applies the identifier returning a list of bounding boxes for each identified shape.

        :param image: The image where shapes will be identified in.
        :type image: PIL.Image.Image
        :return: A list of bounding boxes where is bounding box is a list of 4 coordinates: minx, miny, maxx, maxy.
        :rtype: list[list]
        """
        labels = self._connected_components.run(image)

        bounding_boxes = []

        # For each unique identified connected component, extract its bounding box
        for label in np.unique(labels):
            # Ignore unlabeled background
            if label < 0:
                continue

            mask = np.argwhere(labels == label)

            minx, miny = np.min(mask, axis=0)
            maxx, maxy = np.max(mask, axis=0)

            bounding_boxes.append([minx, miny, maxx, maxy])

        return bounding_boxes


class _ShapeClassifier:
    """
    Implements a shape classifier based on contour approximation.

    This class has been replaced by the template-based classifier
    but is being kept here for historical purposes.
    """
    # These thresholds were chose arbitrarily after empirical experimentation
    # Percentage of the original perimeter to keep
    _PERCENTAGE_OF_ORIGINAL_PERIMETER = 0.015
    # Number of maximum points to resample for the contour
    _RESAMPLING_MAX = 160
    # Percentage of the original perimeter to resample
    _RESAMPLING_PERCENTAGE = 0.5
    # Distance between points to merge them together
    _MERGE_DISTANCE = 10

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
        resampled_contour = _ShapeClassifier._resample(
            contour,
            perimeter,
            max(_ShapeClassifier._RESAMPLING_MAX, int(_ShapeClassifier._RESAMPLING_PERCENTAGE * perimeter))
        )
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

            if distance <= _ShapeClassifier._MERGE_DISTANCE:
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
                if neighbor not in visited and _ContourAreaFinder._is_valid(neighbor):
                    queue.append(neighbor)

        return np.array(area)

    @staticmethod
    def _is_valid(neighbor):
        x, y = neighbor
        return 0 <= x < 184 and 0 <= y < 184


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


class _ConnectedComponentsAlgorithm:
    """
    Implements the connected components algorithm to label shapes inside an image.

    Reference: https://en.wikipedia.org/wiki/Connected-component_labeling
    """
    _WHITE = 1
    _BLACK = 0

    CONNECTIVITY_FOUR = 0
    CONNECTIVITY_EIGHT = 1

    _NEIGHBORS = {
        # Up and left (behind)
        CONNECTIVITY_FOUR: [(0, -1), (-1, 0)],
        # Up-right, up, up-left, left
        CONNECTIVITY_EIGHT: [(1, -1), (0, -1), (-1, -1), (-1, 0)]
    }

    _UNASSIGNED = -1

    def __init__(self, connectivity):
        self._connectivity = connectivity

    def run(self, image):
        """
        Runs the algorithm on the given image returning an array of the same size with the labels.

        :param image: The image to run the connected components algorithm against.
        :type image: PIL.Image.Image
        :return: An array of the same size as `image` with the labels
        :rtype: ndarray
        """
        # Convert each image into a black and white bi-level representation using a custom threshold since Pillow
        # dithers the image adding noise to it. A threshold of 60 was used because only the darkest contours should
        # be kept which helps separating shapes that are "joined" by some lighter grayish pixels that to the human eye
        # are white, but to the image processing library they are still darkish which affects the algorithm.
        # So, anything that is not darker than 60 intensity value, i.e. very black, is considered white.
        # 60 was chosen arbitrarily after empirical experimentation
        # Reference: https://stackoverflow.com/a/50090612
        binary_image = np.array(image.copy().point(lambda x: 1 if x > 60 else 0))
        labeled_image = np.full_like(binary_image, self._UNASSIGNED, dtype=np.int)

        label = 0
        disjoint_set = _DisjointSet()

        # First pass: assign labels
        for i in range(0, binary_image.shape[0]):
            for j in range(0, binary_image.shape[1]):
                # If the current pixel is not background
                if binary_image[i, j] != self._WHITE:
                    # Get the connected neighbors
                    neighbors = self._get_neighbors(i, j, binary_image, labeled_image)

                    # If neighbors is empty
                    if len(neighbors) == 0:
                        # Create a new set for the current label, assign it to the pixel and increment it
                        disjoint_set.make_set(label)
                        labeled_image[i, j] = label
                        label += 1
                    else:
                        # Find the smallest label of all the neighbors
                        neighbors_labels = [labeled_image[nx, ny] for nx, ny in neighbors]
                        labeled_image[i, j] = min(neighbors_labels)

                        # Add equivalence labels
                        for neighbor_label in neighbors_labels:
                            disjoint_set.union(labeled_image[i, j], neighbor_label)

        # Second pass: consolidate equivalence labels
        for i in range(0, binary_image.shape[0]):
            for j in range(0, binary_image.shape[1]):
                if binary_image[i, j] != self._WHITE:
                    labeled_image[i, j] = disjoint_set.find(labeled_image[i, j])

        return labeled_image

    def _get_neighbors(self, px, py, binary_image, labeled_image):
        neighbors = []

        for nx, ny in self._NEIGHBORS[self._connectivity]:
            nx, ny = px + nx, py + ny

            # Filter out neighbors outside of the image
            if nx < 0 or nx >= binary_image.shape[0] or ny < 0 or ny >= binary_image.shape[1]:
                continue

            # Filter out neighbors that are not connected by the current pixel's value
            # or whose labels have not been assigned yet
            if binary_image[nx, ny] != binary_image[px, py] or labeled_image[nx, ny] == self._UNASSIGNED:
                continue

            neighbors.append((nx, ny))

        return neighbors


class _DisjointSet:
    """
    Implements a disjoint set data structure.

    Reference: https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    """

    def __init__(self):
        self._parents = {}
        self._ranks = {}

    def make_set(self, x):
        """
        Makes a new set for the given element.

        :param x: The element to create a new set for.
        """
        self._parents[x] = x
        self._ranks[x] = 0

    def find(self, x):
        """
        Finds the representative, i.e. root, element of the set where `x` belongs.
        It uses a path compression heuristic.

        :param x: The element to find its representative set for.
        :return: The representative set.
        """
        if self._parents[x] != x:
            self._parents[x] = self.find(self._parents[x])

        return self._parents[x]

    def union(self, x, y):
        """
        Merges the representative sets for the given two elements.
        It uses a union by rank heuristic.

        :param x: The first element.
        :param y: The second element.
        """
        x_root = self.find(x)
        y_root = self.find(y)

        # x and y are already in the same set
        if x_root == y_root:
            return

        # x and y are not in same set, so we merge them
        if self._ranks[x_root] < self._ranks[y_root]:
            # Swap x_root and y_root
            x_root, y_root = y_root, x_root

        # Merge y_root into x_root
        self._parents[y_root] = x_root

        if self._ranks[x_root] == self._ranks[y_root]:
            self._ranks[x_root] += 1

    def __repr__(self):
        return 'Parents={}'.format(self._parents)


def _perimeters_are_similar(p1, p2):
    """
    Determines whether two perimeters are similar to each other or not.

    :param p1: The first perimeter.
    :type p1: np.ndarray
    :param p2: The second perimeter.
    :type p2: np.ndarray
    :return: True if the perimeters are similar to each other, False otherwise.
    :rtype: bool
    """
    # The difference threshold was chosen arbitrarily after some empirical experimentation
    return abs(p1 - p2) < 50
