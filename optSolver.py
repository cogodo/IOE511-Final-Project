import numpy as np
from algorithms.algorithms import gradient_descent, newton
from algorithms.base import SolverAlgorithm
from objectives.base import SolverObjective
from options.base import SolverOptions
from objectives.functions import rosen_func, rosen_grad, rosen_Hess, quadratic_func, quadratic_grad, quadratic_Hess

def setProblem(problem: SolverObjective):

    name = problem.name
    if name == 'Rosenbrock':
        problem.value = rosen_func
        problem.grad = rosen_grad
        problem.hess = rosen_Hess
    elif name == 'Quadratic':
        A = problem.A
        b = problem.b
        c = problem.c
        problem.value = lambda x, _A=A, _b=b, _c=c: quadratic_func(A=_A, b=_b, c=_c, x=x)
        problem.grad = lambda x, _A=A, _b=b: quadratic_grad(A=_A, b=_b, x=x)
        problem.hess = lambda x, _A=A: quadratic_Hess(A=_A)
    else:
        raise ValueError("Problem name does not exist!")

    return problem
        
def setMethod(method: SolverAlgorithm):

    def _gradient_descent_step(x, f, g, H, objective, options):
        return gradient_descent(x=x, f=f, g=g, objective=objective, options=options)

    name = method.name
    if name == 'GradientDescent':
        method.step = _gradient_descent_step
    elif name == 'Newton':
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

    _abs = np.abs
    _max = np.max
    norm_g = _max(_abs(g))
    norm_g_x0 = norm_g

    # set initial iteration counter
    k = 0

    # precompute loop-invariant termination threshold
    term_tol = options.term_tol
    max_iterations = options.max_iterations
    term_threshold = term_tol * max(norm_g_x0, 1)

    # cache method.step locally
    _step = method.step

    # 2 types of termination conditions
    while norm_g > term_threshold and k < max_iterations:

        # take a step in the method
        results = _step(x, f, g, H, problem, options)

        # update function values
        x = results.x_new
        f = results.f_new
        g = results.g_new
        norm_g = _max(_abs(g))
        H = results.H_new

        # increment iteration count
        k += 1

    return x, f


    