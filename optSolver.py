import numpy as np
from algorithms.algorithms import gradient_descent, newton, bfgs, dbfgs, cbfgs, lbfgs, dfp, trnewtoncg, trsr1cg, \
    dlbfgs, ddbfgs
from algorithms.base import SolverAlgorithm
from algorithms.utils import VectorCircularBuffer, LBFGSState
from objectives.base import SolverObjective
from options.base import SolverOptions
from objectives.functions import rosen_2_func, rosen_2_grad, rosen_2_Hess,\
                                rosen_100_func, rosen_100_grad, rosen_100_Hess,\
                                quadratic_func, quadratic_grad, quadratic_Hess

def setProblem(problem: SolverObjective):
    
    match problem.name:
        case 'Rosenbrock-2':
            problem.value = rosen_2_func
            problem.grad = rosen_2_grad
            problem.hess = rosen_2_Hess

        case 'Rosenbrock-100':
            problem.value = rosen_100_func
            problem.grad = rosen_100_grad
            problem.hess = rosen_100_Hess

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
            method.step = lambda x, f, g, H, Hinv_approx, H_approx, delta, internal_state, objective, options: gradient_descent(x=x, f=f, g=g, objective=objective, options=options)
        case 'Newton':
            method.step = lambda x, f, g, H, Hinv_approx, H_approx, delta, internal_state, objective, options: newton(x=x, f=f, g=g, H=H, objective=objective, options=options)
        case 'TR-Newton-CG':
            method.step = lambda x, f, g, H, Hinv_approx, H_approx, delta, internal_state, objective, options: trnewtoncg(x=x, f=f, g=g, H=H, delta=delta, objective=objective, options=options)
        case 'TR-SR1-CG':
            method.step = lambda x, f, g, H, Hinv_approx, H_approx, delta, internal_state, objective, options: trsr1cg(x=x, f=f, g=g, H_approx=H_approx, delta=delta, objective=objective, options=options)
        case 'BFGS':
            method.step = lambda x, f, g, H, Hinv_approx, H_approx, delta, internal_state, objective, options: bfgs(x=x, f=f, g=g, Hinv_approx=Hinv_approx, objective=objective, options=options)
        case 'D-BFGS':
            method.step = lambda x, f, g, H, Hinv_approx, H_approx, delta, internal_state, objective, options: dbfgs(x=x, f=f, g=g, Hinv_approx=Hinv_approx, objective=objective, options=options)
        case 'DD-BFGS':
            method.step = lambda x, f, g, H, Hinv_approx, H_approx, delta, internal_state, objective, options: ddbfgs(x=x, f=f, g=g, Hinv_approx=Hinv_approx, objective=objective, options=options)
        case 'C-BFGS':
            method.step = lambda x, f, g, H, Hinv_approx, H_approx, delta, internal_state, objective, options: cbfgs(x=x, f=f, g=g, Hinv_approx=Hinv_approx, objective=objective, options=options)
        case 'L-BFGS':
            method.step = lambda x, f, g, H, Hinv_approx, H_approx, delta, internal_state, objective, options: lbfgs(x=x, f=f, g=g, internal_state=internal_state, objective=objective, options=options)
        case 'DFP':
            method.step = lambda x, f, g, H, Hinv_approx, H_approx, delta, internal_state, objective, options: dfp(x=x, f=f, g=g, Hinv_approx=Hinv_approx, objective=objective, options=options)
        case 'D-L-BFGS':
            method.step = lambda x, f, g, H, Hinv_approx, H_approx, delta, internal_state, objective, options: dlbfgs(x=x, f=f, g=g, internal_state=internal_state, objective=objective, options=options)
        case _:
            raise ValueError("Method name does not exist!")
    return method

def setOptions(options: SolverOptions):
    pass
    return options

def optSolver(problem: SolverObjective, method: SolverAlgorithm, options: SolverOptions, f_vals: list[float]=None):

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
    H_approx = Hinv_approx
    delta = options.trust_region.delta_init
    internal_state = None
    norm_g = np.linalg.norm(g, ord=np.inf)
    norm_g_x0 = norm_g

    # set up initial Hessian guesses for bfgs variants
    if options.bfgs.Hinv_approx_init is not None:
        Hinv_approx = options.bfgs.Hinv_approx_init

    if options.bfgs.H_approx_init is not None:
        H_approx = options.bfgs.H_approx_init

    # set up internal state and initial Hessian approximation for each iteration
    if method.name == 'L-BFGS' or method.name == 'D-L-BFGS':
        internal_state = LBFGSState(s_buffer=VectorCircularBuffer(capacity=options.bfgs.history_length, vector_size=x.size, dtype=x.dtype),
                                    y_buffer=VectorCircularBuffer(capacity=options.bfgs.history_length, vector_size=x.size, dtype=x.dtype))

    # set initial iteration counter
    k = 0

    # 2 types of termination conditions: checking that gradient is small enough and bounding the max number of iterations k
    while not (norm_g <= options.term_tol*max(norm_g_x0, 1) or k >= options.max_iterations):

        # take a step in the method
        results = method.step(x, f, g, H, Hinv_approx, H_approx, delta, internal_state, problem, options)

        # for plotting purposes
        if f_vals is not None:
            f_vals.append(f)

        # update function values
        x = results.x_new
        f = results.f_new
        g = results.g_new
        H = results.H_new
        Hinv_approx = results.Hinv_approx_new
        H_approx = results.H_approx_new
        delta = results.delta_new
        internal_state = results.internal_state
        norm_g = np.linalg.norm(g, ord=np.inf)

        # increment iteration count
        k = k + 1

    return x, f


    