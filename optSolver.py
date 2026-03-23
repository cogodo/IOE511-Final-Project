import numpy as np
from algorithms.algorithms import gradient_descent, newton
from algorithms.base import SolverAlgorithm
from algorithms.utils import SolverOptions
from objectives.base import SolverObjective
from objectives.functions import rosen_func, rosen_grad, rosen_Hess, quadratic_func, quadratic_grad, quadratic_Hess

def setProblem(problem: SolverObjective):
    
    match problem.name:
        case 'Rosenbrock':
            problem.value = rosen_func
            problem.grad = rosen_grad
            problem.hess = rosen_Hess

        case 'Quadratic':
            problem.value = lambda x: quadratic_func(problem.A, problem.b, problem.c, x)
            problem.grad = lambda x: quadratic_grad(problem.A, problem.b, x)
            problem.hess = lambda x: quadratic_Hess(problem.A)

        case _:
            raise ValueError("Problem name does not exist!")
        
    return problem
        
def setMethod(method: SolverAlgorithm):
    pass
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
        match method.name:
            case 'GradientDescent':
                x_new, f_new, g_new, d, alpha = gradient_descent(x=x, f=f, g=g, objective=problem, algorithm=method, options=options)

            case 'Newton':
                x_new, f_new, g_new, H_new, d, alpha = newton(x=x, f=f, g=g, H=H, objective=problem, algorithm=method, options=options)

            case _:
                raise ValueError("Method not found!")
            
        # update function values
        x = x_new
        f = f_new
        g = g_new
        norm_g = np.linalg.norm(g, ord=np.inf)
        H = H_new

        # increment iteration count
        k = k + 1

    return x, f


    