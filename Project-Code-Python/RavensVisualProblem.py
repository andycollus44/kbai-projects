from PIL import Image


class RavensVisualProblemFactory:
    def __init__(self):
        pass

    def create(self, problem_type, figures):
        """
        Creates a RavensVisualProblem instance.

        :param problem_type: The type of the problem, either '2x2' or '3x3'.
        :type problem_type: str
        :param figures: The list of input figures.
        :type figures: list[RavensFigure.RavensFigure]
        :return: A RavensVisualProblem instance.
        :rtype: RavensVisualProblem
        """
        if problem_type == '2x2':
            dimension = 2
            letters = [['A', 'B'], 'C']
            numbers = [str(i) for i in range(1, 7)]
        elif problem_type == '3x3':
            dimension = 3
            letters = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H']]
            numbers = [str(i) for i in range(1, 9)]
        else:
            raise ValueError('Invalid problem type: {}'.format(problem_type))

        matrix = [[self._process_image(figures[col].visualFilename) for col in row] for row in letters]
        answers = [self._process_image(figures[number].visualFilename) for number in numbers]

        return RavensVisualProblem(dimension, matrix, answers)

    def _process_image(self, image_file):
        # Perform the following transformations to the image:
        # 1. Convert to grayscale
        # 2. Reduce size of the image to 32 x 32 so that operations are faster
        return Image.open(image_file).convert('L').resize((32, 32), resample=Image.BICUBIC)


class RavensVisualProblem:
    def __init__(self, dimension, matrix, answers):
        self._dimension = dimension
        self._matrix = matrix
        self._answers = answers

    @property
    def dimension(self):
        return self._dimension

    @property
    def matrix(self):
        return self._matrix

    @property
    def answers(self):
        return self._answers
