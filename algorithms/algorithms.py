import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

from algorithms.utils import StepResults, backtracking_line_search, weak_wolfe_line_search
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
            alpha = backtracking_line_search(x=x, f=f, g=g, d=d, objective=objective, options=options)

        case 'Wolfe':
            alpha = weak_wolfe_line_search(x=x, f=f, g=g, d=d, objective=objective, options=options)

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
            alpha = backtracking_line_search(x=x, f=f, g=g, d=d, objective=objective, options=options)

        case 'Wolfe':
            alpha = weak_wolfe_line_search(x=x, f=f, g=g, d=d, objective=objective, options=options)
                
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

def bfgs(x: Array, f: Array, g: Array, Hinv_approx: Array, objective: SolverObjective, options: SolverOptions):
    
    # search direction is the Newton direction, but with the inverse Hessian approximation
    d = -Hinv_approx @ g




def dfp(objective: SolverObjective, x: Array, options: SolverOptions):
    pass