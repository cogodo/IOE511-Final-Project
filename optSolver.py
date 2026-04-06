import numpy as np
from algorithms.algorithms import gradient_descent, newton, bfgs, lbfgs, dfp
from algorithms.base import SolverAlgorithm
from algorithms.utils import VectorCircularBuffer, LBFGSState
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
            method.step = lambda x, f, g, H, Hinv_approx, internal_state, objective, options: gradient_descent(x=x, f=f, g=g, objective=objective, options=options)
        case 'Newton':
            method.step = lambda x, f, g, H, Hinv_approx, internal_state, objective, options: newton(x=x, f=f, g=g, H=H, objective=objective, options=options)
        case 'BFGS':
            method.step = lambda x, f, g, H, Hinv_approx, internal_state, objective, options: bfgs(x=x, f=f, g=g, Hinv_approx=Hinv_approx, objective=objective, options=options)
        case 'L-BFGS':
            method.step = lambda x, f, g, H, Hinv_approx, internal_state, objective, options: lbfgs(x=x, f=f, g=g, internal_state=internal_state, objective=objective, options=options)
        case 'DFP':
            method.step = lambda x, f, g, H, Hinv_approx, internal_state, objective, options: dfp(x=x, f=f, g=g, Hinv_approx=Hinv_approx, objective=objective, options=options)
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
    internal_state = None
    norm_g = np.linalg.norm(g, ord=np.inf)
    norm_g_x0 = norm_g

    # set up initial Hessian guesses for bfgs variants
    if options.bfgs.Hinv_approx_init is not None:
        Hinv_approx = options.bfgs.Hinv_approx_init

    # set up internal state and initial Hessian approximation for each iteration
    if method.name == 'L-BFGS':
        internal_state = LBFGSState(s_buffer=VectorCircularBuffer(capacity=options.bfgs.history_length, vector_size=x.size, dtype=x.dtype),
                                    y_buffer=VectorCircularBuffer(capacity=options.bfgs.history_length, vector_size=x.size, dtype=x.dtype))

    # set initial iteration counter
    k = 0

    # 2 types of termination conditions: checking that gradient is small enough and bounding the max number of iterations k
    while not (norm_g <= options.term_tol*max(norm_g_x0, 1) or k >= options.max_iterations):

        # take a step in the method
        results = method.step(x, f, g, H, Hinv_approx, internal_state, problem, options)

        # update function values
        x = results.x_new
        f = results.f_new
        g = results.g_new
        H = results.H_new
        Hinv_approx = results.Hinv_approx_new
        internal_state = results.internal_state
        norm_g = np.linalg.norm(g, ord=np.inf)

        # increment iteration count
        k = k + 1

    return x, f


    