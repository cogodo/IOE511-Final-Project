import numpy as np
from algorithms.algorithms import gradient_descent, newton
from algorithms.base import SolverAlgorithm
from objectives.base import SolverObjective
from options.base import SolverOptions
from objectives.functions import rosen_func, rosen_grad, rosen_Hess, quadratic_func, quadratic_grad, quadratic_Hess

def setProblem(problem: SolverObjective):

    match problem.name:
        case 'Rosenbrock':
            problem.value = rosen_func
            problem.grad = rosen_grad
            problem.hess = rosen_Hess

        case 'Quadratic':
            problem.value = lambda x: quadratic_func(A=problem.A, b=problem.b, c=problem.c, x=x)
            problem.grad = lambda x: quadratic_grad(A=problem.A, b=problem.b, x=x)
            problem.hess = lambda x: quadratic_Hess(A=problem.A)

        case _:
            raise ValueError("Problem name does not exist!")

    return problem
        
def setMethod(method: SolverAlgorithm):

    # set the step for every iteration
    match method.name:
        case 'GradientDescent':
            def _gradient_descent_wrapper(x, f, g, H, objective, options):
                return gradient_descent(x=x, f=f, g=g, objective=objective, options=options)
            method.step = _gradient_descent_wrapper
        case 'Newton':
            method.step = newton
    return method

def setOptions(options: SolverOptions):
    return options

def optSolver(problem: SolverObjective, method: SolverAlgorithm, options: SolverOptions):

    # set problem, method, and options
    problem = setProblem(problem)
    method = setMethod(method)
    options = setOptions(options)

    # compute initial function/gradient/Hessian
    x = problem.x0
    f = problem.value(x)
    g = problem.grad(x)
    H = problem.hess(x)
    _norm = np.linalg.norm
    _inf = np.inf
    norm_g = _norm(g, ord=_inf)
    norm_g_x0 = norm_g

    # set initial iteration counter
    k = 0

    # precompute invariant threshold and max iterations
    tol_threshold = options.term_tol * max(norm_g_x0, 1)
    max_iter = options.max_iterations

    # cache method.step locally
    _step = method.step

    # 2 types of termination conditions: checking that gradient is small enough and bounding the max number of iterations k
    while norm_g > tol_threshold and k < max_iter:

        # take a step in the method
        results = _step(x, f, g, H, problem, options)

        # update function values
        x = results.x_new
        f = results.f_new
        g = results.g_new
        norm_g = _norm(g, ord=_inf)
        H = results.H_new

        # increment iteration count
        k += 1

    return x, f


    