import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

from algorithms.utils import StepResults, backtracking_line_search, weak_wolfe_line_search, two_loop_recursion, cg, eval_m_k, LBFGSState
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

def trnewtoncg(x: Array, f: Array, g: Array, H: Array, delta: float, objective: SolverObjective, options: SolverOptions):

    # direction is the solution to the TR subproblem with B_k equal to the exact Hessian
    d = cg(f=f, g=g, B=H, delta=delta, options=options)
    rho = (f - objective.value(x + d)) / (eval_m_k(f_k=f, g_k=g, B_k=H, d=np.zeros(np.shape(x))) - eval_m_k(f_k=f, g_k=g, B_k=H, d=d))

    x_new = x
    delta_new = delta

    if rho > options.trust_region.c1:
        x_new = x + d
        if rho > options.trust_region.c2:
            delta_new = 2 * delta
    else:
        delta_new = 0.5 * delta

    results = StepResults(x_new=x_new,
                          f_new=objective.value(x_new),
                          g_new=objective.grad(x_new),
                          H_new=objective.hess(x_new),
                          delta_new=delta_new,
                          d=d)
    
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

def dbfgs(x: Array, f: Array, g: Array, Hinv_approx: Array, H_approx: Array, objective: SolverObjective, options: SolverOptions):

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

    s_k = x_new - x
    y_k = g_new - g

    # instead of skipping the update when the sy tolerance is not met, we will set theta to interpolate current inverse Hessian approximation and the one produced by the BFGS formula
    theta_k = 1
    if (s_k.transpose() @ y_k < 0.2*s_k.transpose() @ H_approx @ s_k):
        theta_k = (0.8*s_k.transpose() @ H_approx @ s_k) / (s_k.transpose() @ H_approx @ s_k - s_k.transpose() @ y_k)

    r_k = theta_k * y_k + (1 - theta_k) * H_approx @ s_k

    # update using BFGS formulas: choice of theta ensures positive definite-ness of update
    H_approx_new = H_approx - (H_approx @ s_k @ s_k.transpose() @ H_approx) / (s_k.transpose() @ H_approx @ s_k) + (r_k @ r_k.transpose()) / (s_k.transpose() @ r_k)
    Hinv_approx_new = (np.eye(np.size(Hinv_approx, 0)) - (s_k @ r_k.transpose()) / (s_k.transpose() @ r_k)) @ Hinv_approx @ (np.eye(np.size(Hinv_approx, 0)) - (r_k @ s_k.transpose()) / (s_k.transpose() @ r_k)) + (s_k @ s_k.transpose()) / (s_k.transpose() @ r_k)

    results = StepResults(x_new=x_new,
                          f_new=f_new,
                          g_new=g_new,
                          Hinv_approx_new=Hinv_approx_new,
                          H_approx_new=H_approx_new,
                          d=d,
                          alpha=alpha)
    
    return results

def cbfgs(x: Array, f: Array, g: Array, Hinv_approx: Array, objective: SolverObjective, options: SolverOptions):

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

    # update the inverse Hessian approximation, only if the CAUTIOUS BFGS tolerance is met
    s_k = x_new - x
    y_k = g_new - g
    
    if y_k.transpose() @ s_k >= options.bfgs.cautious_tol * pow(np.linalg.norm(g), options.bfgs.cautious_alpha) * pow(np.linalg.norm(s_k), 2):
        Hinv_approx_new = (np.eye(np.size(Hinv_approx, 0)) - (s_k @ y_k.transpose()) / (s_k.transpose() @ y_k)) @ Hinv_approx @ (np.eye(np.size(Hinv_approx, 0)) - (y_k @ s_k.transpose()) / (s_k.transpose() @ y_k)) + (s_k @ s_k.transpose()) / (s_k.transpose() @ y_k)
    
    results = StepResults(x_new=x_new,
                          f_new=f_new,
                          g_new=g_new,
                          Hinv_approx_new=Hinv_approx_new,
                          d=d,
                          alpha=alpha)
    
    return results

def lbfgs(x: Array, f: Array, g: Array, internal_state: LBFGSState, objective: SolverObjective, options: SolverOptions):

    Hinv_approx_init = np.eye(np.size(x, 0))
    if options.bfgs.Hinv_approx_init is not None:
        Hinv_approx_init = options.bfgs.Hinv_approx_init

    d = two_loop_recursion(g=g, Hinv_approx_init=Hinv_approx_init, s_buffer=internal_state.s_buffer, y_buffer=internal_state.y_buffer)

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

    # update the internal state, only if sy tolerance is met
    s_k = x_new - x
    y_k = g_new - g

    if s_k.transpose() @ y_k >= options.bfgs.sy_tol * np.linalg.norm(s_k) * np.linalg.norm(y_k):
        internal_state.s_buffer.append(np.squeeze(s_k))
        internal_state.y_buffer.append(np.squeeze(y_k))

    results = StepResults(x_new=x_new,
                          f_new=f_new,
                          g_new=g_new,
                          d=d,
                          alpha=alpha,
                          internal_state=internal_state)

    return results

def dfp(x: Array, f: Array, g: Array, Hinv_approx: Array, objective: SolverObjective, options: SolverOptions):

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
        Hinv_approx_new = Hinv_approx - (Hinv_approx @ y_k @ y_k.transpose() @ Hinv_approx) / (y_k.transpose() @ Hinv_approx @ y_k) + (s_k @ s_k.transpose()) / (s_k.transpose() @ y_k)
    
    results = StepResults(x_new=x_new,
                          f_new=f_new,
                          g_new=g_new,
                          Hinv_approx_new=Hinv_approx_new,
                          d=d,
                          alpha=alpha)
    
    return results