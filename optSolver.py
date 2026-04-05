import numpy as np
from algorithms.algorithms import gradient_descent, newton, bfgs, lbfgs
from algorithms.base import SolverAlgorithm
from algorithms.utils import VectorCircularBuffer, BFGSState, InternalAlgorithmState
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
    
    # set the step for every iteratiokn
    match method.name:
        case 'GradientDescent':
            method.step = lambda x, f, g, H, Hinv_approx, objective, options: gradient_descent(x=x, f=f, g=g, objective=objective, options=options)
        case 'Newton':
            method.step = lambda x, f, g, H, Hinv_approx, objective, options: newton(x=x, f=f, g=g, H=H, objective=objective, options=options)
        case 'BFGS':
            method.step = lambda x, f, g, H, Hinv_approx, objective, options: bfgs(x=x, f=f, g=g, Hinv_approx=Hinv_approx, objective=objective, options=options)
        case 'L-BFGS':
            method.step = lambda x, f, g, H, Hinv_approx, s_buffer, y_buffer, objective, options: lbfgs(x=x, f=f, g=g, Hinv_approx=Hinv_approx, s_buffer=s_buffer, y_buffer = y_buffer,objective=objective, options=options)
        case _:
            raise ValueError("Method name does not exist!")
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
    Hinv_approx = np.eye(np.size(H, 0))
    norm_g = np.linalg.norm(g, ord=np.inf)
    norm_g_x0 = norm_g

    match method.name:
        case 'L-BFGS':
            s_buffer = VectorCircularBuffer(capacity=options.lbfgs.history_length, vector_size=x.size, dtype=x.dtype)
            y_buffer = VectorCircularBuffer(capacity=options.lbfgs.history_length, vector_size=x.size, dtype=x.dtype)


    # set initial iteration counter
    k = 0

    # 2 types of termination conditions: checking that gradient is small enough and bounding the max number of iterations k
    while not (norm_g <= options.term_tol*max(norm_g_x0, 1) or k >= options.max_iterations):

        # take a step in the method
        match method.name:
            case 'L-BFGS':
                results = method.step(x, f, g, H, Hinv_approx, s_buffer, y_buffer, problem, options)
                lbfgs_internal_state = results.internal_state
                s_buffer = lbfgs_internal_state.s_buffer
                y_buffer = lbfgs_internal_state.y_buffer
            case _:
                results = method.step(x, f, g, H, Hinv_approx, problem, options)

        # update function values
        x = results.x_new
        f = results.f_new
        g = results.g_new
        norm_g = np.linalg.norm(g, ord=np.inf)
        H = results.H_new
        Hinv_approx = results.Hinv_approx_new

        # increment iteration count
        k = k + 1

    return x, f


    