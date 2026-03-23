import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

from algorithms.utils import SolverOptions
from algorithms.base import SolverAlgorithm
from objectives.base import SolverObjective

def gradient_descent(x: Array, f: float, g: Array, objective: SolverObjective, algorithm: SolverAlgorithm, options: SolverOptions):
    
    # search direction is -g
    d = -g

    # determine the step size
    alpha = 0
    match algorithm.line_search.method:
        case 'Constant':
            alpha = algorithm.line_search.const_alpha
        case 'Backtracking':
            alpha = algorithm.line_search.alpha0

            # perform backtracking line search
            while objective.value(x + alpha*d) > f + algorithm.line_search.c1*alpha*g.transpose() @ d:
                alpha = alpha*algorithm.line_search.tau

    x_new = x + alpha*d
    f_new = objective.value(x_new)
    g_new = objective.grad(x_new)

    return (x_new, f_new, g_new, d, alpha)


def newton(objective: SolverObjective, x: Array, options: SolverOptions):
    pass

def trsr1cg(objective: SolverObjective, x: Array, options: SolverOptions):
    pass

def sr1(objective: SolverObjective, x: Array, options: SolverOptions):
    pass

def bfgs(objective: SolverObjective, x: Array, options: SolverOptions):
    pass

def dfp(objective: SolverObjective, x: Array, options: SolverOptions):
    pass