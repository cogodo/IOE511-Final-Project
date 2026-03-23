import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

from algorithms.utils import StepResults
from objectives.base import SolverObjective
from options.base import SolverOptions

def gradient_descent(x: Array, f: float, g: Array, objective: SolverObjective, options: SolverOptions):
    
    # search direction is -g
    d = -g

    # determine the step size
    alpha = 0
    match options.line_search.method:
        case 'Constant':
            alpha = options.line_search.const_alpha
        case 'Backtracking':
            alpha = options.line_search.alpha0

            # perform backtracking line search
            while objective.value(x + alpha*d) > f + options.line_search.c1*alpha*g.transpose() @ d:
                alpha = alpha*options.line_search.tau

        case 'Wolfe':
            pass

    x_new = x + alpha*d

    results = StepResults(x_new=x_new,
                          f_new=objective.value(x_new),
                          g_new=objective.grad(x_new),
                          d=d,
                          alpha=alpha)
    
    return results


def newton(x: Array, f: float, g: Array, H: Array, objective: SolverObjective, options: SolverOptions):

    # search direction is -inv(H) * g
    d = -np.linalg.inv(H) @ g
    
    # determine the step size
    alpha = 0
    match options.line_search.method:
        case 'Backtracking':
            alpha = options.line_search.alpha0

            # perform backtracking line search
            while objective.value(x + alpha*d) > f + options.line_search.c1*alpha*g.transpose() @ d:
                alpha = alpha*options.line_search.tau
        case 'Wolfe':
            pass

    x_new = x + alpha*d

    results = StepResults(x_new=x_new,
                          f_new=objective.value(x_new),
                          g_new=objective.grad(x_new),
                          H_new=objective.hess(x_new),
                          d=d,
                          alpha=alpha)

    return results

def trsr1cg(objective: SolverObjective, x: Array, options: SolverOptions):
    pass

def sr1(objective: SolverObjective, x: Array, options: SolverOptions):
    pass

def bfgs(objective: SolverObjective, x: Array, options: SolverOptions):
    pass

def dfp(objective: SolverObjective, x: Array, options: SolverOptions):
    pass