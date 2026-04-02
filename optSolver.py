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
        
from functools import partial


def gradient_descent_wrapper(x, f, g, H, objective, options):
    return gradient_descent(x=x, f=f, g=g, objective=objective, options=options)


def setMethod(method: SolverAlgorithm):

    # set the step for every iteratiokn
    match method.name:
        case 'GradientDescent':
            method.step = lambda x, f, g, H, objective, options: gradient_descent(x=x, f=f, g=g, objective=objective, options=options)
        case 'Newton':
            method.step = newton
    return method

def setOptions(options: SolverOptions):
    pass
    return options

def optSolver(problem: SolverObjective, method: SolverAlgorithm, options: SolverOptions):

    # set problem, method, and options
    problem = setProblem(problem)
    method = setMethod(method)
    options = setOptions(options)

    # compute initial function/gradient/Hessian
    x = problem.x0
    p_value = problem.value
    p_grad = problem.grad
    p_hess = problem.hess
    m_step = method.step

    f = p_value(x)
    g = p_grad(x)
    H = p_hess(x)

    _abs = np.abs
    _max = np.max

    norm_g = _max(_abs(g))
    norm_g_x0 = norm_g

    # precompute termination threshold
    term_threshold = options.term_tol * max(norm_g_x0, 1)
    max_iterations = options.max_iterations

    # set initial iteration counter
    k = 0

    # 2 types of termination conditions
    while norm_g > term_threshold and k < max_iterations:

        # take a step in the method
        results = m_step(x, f, g, H, problem, options)

        # update function values
        x = results.x_new
        f = results.f_new
        g = results.g_new
        norm_g = _max(_abs(g))
        H = results.H_new

        # increment iteration count
        k += 1

    return x, f


    