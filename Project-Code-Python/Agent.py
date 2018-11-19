# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

from RavensProblemSolver import RavensProblemSolverFactory
from RavensSemanticSolver import RavensSemanticSolverFactory
from RavensVisualProblem import RavensVisualProblemFactory

# A set of problems I do not plan to attempt
_BLACKLISTED_PROBLEMS = {
    # Unfortunately, Challenge Problem E-12 causes an infinite loop somewhere in the shape extractor
    'Challenge Problem E-12'
}


class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        self._visual_solver_factory = RavensProblemSolverFactory()
        self._semantic_solver_factory = RavensSemanticSolverFactory()
        self._problem_factory = RavensVisualProblemFactory()

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self, problem):
        # If the problem is blacklisted, skip it!
        if problem.name in _BLACKLISTED_PROBLEMS:
            print 'Skipping problem "{}"'.format(problem.name)
            return -1

        visual_solver = self._visual_solver_factory.create(problem.problemType)
        semantic_solver = self._semantic_solver_factory.create(problem.problemType)
        visual_problem = self._problem_factory.create(problem.problemType, problem.figures)

        # Attempt to solve the problem visually first, if no answer is found then try the semantic approach, else skip
        answer = visual_solver.run(visual_problem) or semantic_solver.run(visual_problem) or -1

        print 'The answer to problem "{}" is {}'.format(problem.name, answer)

        return answer
