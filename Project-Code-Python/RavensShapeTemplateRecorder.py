"""
This script allows the collection of shape templates by extracting shapes from
the given problems and then asking the user to input the name of the shape
in order for it to be converted into the template for that shape.
"""

from PIL import Image, ImageDraw

from ProblemSet import ProblemSet
from RavensShape import RavensShapeExtractor, RavensShapeTemplateClassifier
from RavensVisualProblem import RavensVisualProblemFactory

_SKIP = 'SKIP'


def main():
    problem_sets = ['Basic Problems B', 'Challenge Problems B']
    extractor = RavensShapeExtractor()
    classifier = RavensShapeTemplateClassifier()
    templates_to_save = []

    print 'Running template recorder for problem sets: {}'.format(problem_sets)

    for problem_set in problem_sets:
        problems = ProblemSet(problem_set).problems

        for problem in problems:
            skip = raw_input('\nSkip problem {}? '.format(problem.name)).upper()

            if skip == 'Y':
                continue

            problem = RavensVisualProblemFactory().create(problem.problemType, problem.figures)

            for answer in problem.answers:
                shapes = extractor.apply(answer)

                for shape in shapes:
                    _show(shape)

                    correct = 'N'
                    name = None

                    while correct != 'Y':
                        name = raw_input('\nWhat shape is this? ').upper()

                        if name == _SKIP:
                            break

                        correct = raw_input('Is {} correct? '.format(name)).upper()

                    if name == _SKIP:
                        continue

                    templates_to_save.append(RavensShapeTemplateClassifier.Template(name=name, points=shape.contour))

    print '\nSaving {} templates'.format(len(templates_to_save))
    classifier.save_templates(templates_to_save)

    print '[DONE]'


def _show(shape):
    image = Image.new('RGB', (184, 184), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.polygon(shape.contour.flatten().tolist(), outline=(0, 0, 0))
    image.show()


if __name__ == '__main__':
    main()
