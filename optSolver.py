from algorithms.base import Algorithm
from algorithms.utils import SolverOptions
from objectives.base import Objective


def optSolver(problem: Objective, method: Algorithm, options: SolverOptions):
    x = problem.x0
    f = problem.value
    g = problem.grad

    pass