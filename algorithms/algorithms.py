import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

from algorithms.utils import StepResults
from objectives.base import SolverObjective
from options.base import SolverOptions

def gradient_descent(x: Array, f: float, g: Array, objective: SolverObjective, options: SolverOptions):

    # search direction is -g
    d = -g

    # Pre-compute g^T @ d once (constant throughout the function)
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
            c1_gtd = c1 * gtd
            val = objective.value(x + alpha * d)
            while val > f + alpha * c1_gtd:
                alpha = alpha * tau
                val = objective.value(x + alpha * d)
            cached_f_new = val

        case 'Wolfe':
            alpha = options.line_search.alpha0
            alpha_low = options.line_search.alpha_low0
            alpha_high = options.line_search.alpha_high0
            c1 = options.line_search.c1
            c2 = options.line_search.c2
            c = options.line_search.c
            c1_gtd = c1 * gtd
            c2_gtd = c2 * gtd

            # perform weak Wolfe line search
            while True:
                x_trial = x + alpha * d
                val = objective.value(x_trial)
                if val <= f + alpha * c1_gtd:
                    grad_trial = objective.grad(x_trial)
                    if grad_trial.transpose() @ d >= c2_gtd:
                        cached_f_new = val
                        cached_g_new = grad_trial
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