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
    
    match method.name:
        case 'GradientDescent':
            method.step = lambda x, f, g, H, objective, options: gradient_descent(x=x, f=f, g=g, objective=objective, options=options)
        case 'Newton':
            method.step = lambda x, f, g, H, objective, options: newton(x=x, f=f, g=g, H=H, objective=objective, options=options)
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
    norm_g = np.linalg.norm(g, ord=np.inf)
    norm_g_x0 = norm_g

    # initialize new variables
    x_new = x
    f_new = f
    g_new = g
    H_new = H

    # set initial iteration counter
    k = 0

    # 2 types of termination conditions: checking that gradient is small enough and bounding the max number of iterations k
    while not (norm_g <= options.term_tol*max(norm_g_x0, 1) or k >= options.max_iterations):

        # take a step in the method
        results = method.step(x, f, g, H, problem, options)
            
        # update function values
        x = results.x_new
        f = results.f_new
        g = results.g_new
        norm_g = np.linalg.norm(g, ord=np.inf)
        H = results.H_new

        # increment iteration count
        k = k + 1

    return x, f


    