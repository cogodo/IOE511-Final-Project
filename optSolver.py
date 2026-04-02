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
            A = problem.A
            b = problem.b
            c = problem.c
            problem.value = lambda x, A=A, b=b, c=c: quadratic_func(A=A, b=b, c=c, x=x)
            problem.grad = lambda x, A=A, b=b: quadratic_grad(A=A, b=b, x=x)
            problem.hess = lambda x, A=A: quadratic_Hess(A=A)

        case _:
            raise ValueError("Problem name does not exist!")

    return problem
        
def setMethod(method):
    name = method.name
    if name == 'GradientDescent':
        method.step = gradient_descent
    elif name == 'Newton':
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
    f = problem.value(x)
    g = problem.grad(x)
    H = problem.hess(x)
    _norm = np.linalg.norm
    norm_g = _norm(g, ord=np.inf)
    norm_g_x0 = norm_g

    # set initial iteration counter
    k = 0

    # precompute termination threshold and max iterations
    term_threshold = options.term_tol * max(norm_g_x0, 1)
    max_iter = options.max_iterations

    # 2 types of termination conditions: checking that gradient is small enough and bounding the max number of iterations k
    while norm_g > term_threshold and k < max_iter:

        # take a step in the method
        results = method.step(x, f, g, H, problem, options)

        # update function values
        x = results.x_new
        f = results.f_new
        g = results.g_new
        norm_g = _norm(g, ord=np.inf)
        H = results.H_new

        # increment iteration count
        k = k + 1

    return x, f


    