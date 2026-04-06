import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

from algorithms.utils import StepResults, backtracking_line_search, weak_wolfe_line_search, VectorCircularBuffer, \
    two_loop_recursion, LBFGSState
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
            raise ValueError("Line search method is invalid!")

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
        n_k = -np.diag(H).min() + options.newton.cholesky_beta

    # attempt Cholesky factorization on H + n_k * I until success
    cholesky_success = False
    while not cholesky_success:
        try:
            _ = np.linalg.cholesky(H + n_k * np.eye(np.size(H, 0)))
            cholesky_success = True

        except np.linalg.LinAlgError:

            # modify n_k upon failure, then try again
            n_k = max(2 * n_k, options.newton.cholesky_beta)

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
            raise ValueError("Line search method is invalid!")

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

    # determine the step size
    alpha = 0
    match options.line_search.method:
        case 'Backtracking':
            alpha = backtracking_line_search(x=x, f=f, g=g, d=d, objective=objective, options=options)
        case 'Wolfe':
            alpha = weak_wolfe_line_search(x=x, f=f, g=g, d=d, objective=objective, options=options)
        case _:
            raise ValueError("Line search method is invalid!")
    
    x_new = x + alpha*d
    f_new = objective.value(x_new)
    g_new = objective.grad(x_new)
    Hinv_approx_new = Hinv_approx

    # update the inverse Hessian approximation, only if sy tolerance is met
    s_k = x_new - x
    y_k = g_new - g
    
    if s_k.transpose() @ y_k >= options.bfgs.sy_tol * np.linalg.norm(s_k) * np.linalg.norm(y_k):
        Hinv_approx_new = (np.eye(np.size(Hinv_approx, 0)) - (s_k @ y_k.transpose()) / (s_k.transpose() @ y_k)) @ Hinv_approx @ (np.eye(np.size(Hinv_approx, 0)) - (y_k @ s_k.transpose()) / (s_k.transpose() @ y_k)) + (s_k @ s_k.transpose()) / (s_k.transpose() @ y_k)
    
    results = StepResults(x_new=x_new,
                          f_new=f_new,
                          g_new=g_new,
                          Hinv_approx_new=Hinv_approx_new,
                          d=d,
                          alpha=alpha)
    
    return results


def lbfgs(x: Array, f: Array, g: Array, Hinv_approx: Array, s_buffer: VectorCircularBuffer, y_buffer: VectorCircularBuffer,objective: SolverObjective, options: SolverOptions):

    d = two_loop_recursion(g, Hinv_approx, s_buffer, y_buffer)

    # determine the step size
    alpha = 0
    match options.line_search.method:
        case 'Backtracking':
            alpha = backtracking_line_search(x=x, f=f, g=g, d=d, objective=objective, options=options)
        case 'Wolfe':
            alpha = weak_wolfe_line_search(x=x, f=f, g=g, d=d, objective=objective, options=options)
        case _:
            raise ValueError("Line search method is invalid!")

    x_new = x + alpha * d
    f_new = objective.value(x_new)
    g_new = objective.grad(x_new)
    Hinv_approx_new = Hinv_approx

    # update the inverse Hessian approximation, only if sy tolerance is met
    s_k = x_new - x
    y_k = g_new - g

    s_buffer.append(np.squeeze(s_k))
    y_buffer.append(np.squeeze(y_k))

    results = StepResults(x_new=x_new,
                          f_new=f_new,
                          g_new=g_new,
                          Hinv_approx_new=Hinv_approx_new,
                          d=d,
                          alpha=alpha,
                          internal_state=LBFGSState(s_buffer=s_buffer, y_buffer=y_buffer))

    return results

def dfp(objective: SolverObjective, x: Array, options: SolverOptions):
    pass