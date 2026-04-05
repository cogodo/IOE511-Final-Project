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
                          d=d,
                          alpha=alpha)
    
    return results


def newton(x: Array, f: float, g: Array, H: Array, objective: SolverObjective, options: SolverOptions):

    # ensure that the Hessian is positive definite, if not, apply Newton modification
    n_k = 0
    if np.diag(H).min() <= 0:
        n_k = -np.diag(H).min() + options.cholesky_beta

    # attempt Cholesky factorization on H + n_k * I until success
    cholesky_success = False
    while not cholesky_success:
        try:
            _ = np.linalg.cholesky(H + n_k * np.eye(np.size(H, 0)))
            cholesky_success = True

        except np.linalg.LinAlgError:

            # modify n_k upon failure, then try again
            n_k = max(2 * n_k, options.cholesky_beta)

    # search direction is -inv(H + n_k * I) * g
    d = -np.linalg.inv(H + n_k * np.eye(np.size(H, 0))) @ g
    
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