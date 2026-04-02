import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

from algorithms.utils import StepResults
from objectives.base import SolverObjective
from options.base import SolverOptions

def gradient_descent(x: Array, f: float, g: Array, objective: SolverObjective, options: SolverOptions):

    # search direction is -g
    d = -g

    # Pre-compute g.transpose() @ d once (since d = -g, this is -||g||^2)
    g_transpose_d = g.transpose() @ d

    # determine the step size
    alpha = 0
    f_new_cached = None
    g_new_cached = None

    ls = options.line_search

    match ls.method:
        case 'Constant':
            alpha = ls.const_alpha
        case 'Backtracking':
            alpha = ls.alpha0
            c1 = ls.c1
            tau = ls.tau

            # perform backtracking line search
            x_trial = x + alpha * d
            f_trial = objective.value(x_trial)
            while f_trial > f + c1 * alpha * g_transpose_d:
                alpha = alpha * tau
                x_trial = x + alpha * d
                f_trial = objective.value(x_trial)

            f_new_cached = f_trial

        case 'Wolfe':
            alpha = ls.alpha0
            alpha_low = ls.alpha_low0
            alpha_high = ls.alpha_high0
            c1 = ls.c1
            c2 = ls.c2
            c_coeff = ls.c
            c2_gtd = c2 * g_transpose_d

            # perform weak Wolfe line search
            while True:
                x_trial = x + alpha * d
                f_trial = objective.value(x_trial)
                if f_trial <= f + c1 * alpha * g_transpose_d:
                    g_trial = objective.grad(x_trial)
                    if g_trial.transpose() @ d >= c2_gtd:
                        f_new_cached = f_trial
                        g_new_cached = g_trial
                        break
                    alpha_low = alpha
                else:
                    alpha_high = alpha

                alpha = c_coeff * alpha_low + (1 - c_coeff) * alpha_high

        case _:
            raise ValueError("Line search method does not exist!")

    x_new = x + alpha * d

    if f_new_cached is None:
        f_new_cached = objective.value(x_new)
    if g_new_cached is None:
        g_new_cached = objective.grad(x_new)

    results = StepResults(x_new=x_new,
                          f_new=f_new_cached,
                          g_new=g_new_cached,
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