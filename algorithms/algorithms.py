import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

from algorithms.utils import StepResults
from objectives.base import SolverObjective
from options.base import SolverOptions

def gradient_descent(x: Array, f: float, g: Array, objective: SolverObjective, options: SolverOptions):

    # search direction is -g
    d = -g

    # precompute g^T @ d as a scalar to avoid repeated matrix operations
    gtd = g.transpose() @ d

    # determine the step size
    alpha = 0
    cached_f_new = None
    cached_g_new = None

    match options.line_search.method:
        case 'Constant':
            alpha = options.line_search.const_alpha
        case 'Backtracking':
            alpha = options.line_search.alpha0
            c1 = options.line_search.c1
            tau = options.line_search.tau

            # perform backtracking line search
            while True:
                x_trial = x + alpha * d
                f_trial = objective.value(x_trial)
                if f_trial <= f + c1 * alpha * gtd:
                    cached_f_new = f_trial
                    break
                alpha = alpha * tau

        case 'Wolfe':
            alpha = options.line_search.alpha0
            alpha_low = options.line_search.alpha_low0
            alpha_high = options.line_search.alpha_high0
            c1 = options.line_search.c1
            c2 = options.line_search.c2
            c = options.line_search.c
            c2_gtd = c2 * gtd

            # perform weak Wolfe line search
            while True:
                x_trial = x + alpha * d
                f_trial = objective.value(x_trial)
                if f_trial <= f + c1 * alpha * gtd:
                    g_trial = objective.grad(x_trial)
                    if g_trial.transpose() @ d >= c2_gtd:
                        cached_f_new = f_trial
                        cached_g_new = g_trial
                        break
                    alpha_low = alpha
                else:
                    alpha_high = alpha

                alpha = c * alpha_low + (1 - c) * alpha_high

        case _:
            raise ValueError("Line search method does not exist!")

    x_new = x + alpha * d

    if cached_f_new is None:
        cached_f_new = objective.value(x_new)
    if cached_g_new is None:
        cached_g_new = objective.grad(x_new)

    results = StepResults(x_new=x_new,
                          f_new=cached_f_new,
                          g_new=cached_g_new,
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
            alpha = options.line_search.alpha0
            alpha_low = options.line_search.alpha_low0
            alpha_high = options.line_search.alpha_high0

            # perform weak Wolfe line search
            while True:
                if (objective.value(x + alpha*d) <= f + options.line_search.c1*alpha*g.transpose() @ d):
                    if (objective.grad(x + alpha*d).transpose() @ d >= options.line_search.c2*g.transpose() @ d):
                        break
                    alpha_low = alpha
                else:
                    alpha_high = alpha
                
                alpha = options.line_search.c*alpha_low + (1 - options.line_search.c)*alpha_high
                
        case _:
            raise ValueError("Line search method does not exist!")

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