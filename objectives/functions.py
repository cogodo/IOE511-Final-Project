# All problems + associated gradients and Hessians
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

def rosen_func(x):
    """ Compute function value for Rosenbrock problem"""
    x = np.asarray(x, dtype=float)
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    """ Compute gradient for Rosenbrock problem"""
    x = np.asarray(x, dtype=float)
    return np.array(
        [
            2 * (-1 + x[0] + 200 * x[0] ** 3 - 200 * x[0] * x[1]),
            200 * (-x[0] ** 2 + x[1]),
        ],
        dtype=float,
    )


def rosen_Hess(x):
    """ Compute Hessian for Rosenbrock problem"""

    x = np.asarray(x, dtype=float)
    return np.array(
        [
            [1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
            [-400 * x[0], 200],
        ],
        dtype=float,
    )

def quadratic_func(A: Array, b: Array, c: Array, x: Array):

    """ Compute function value for quadratic problems"""

    return 0.5*(x.transpose() @ A @ x) + b.transpose() @ x + c

def quadratic_grad(A: Array, b: Array, x: Array):

    """ Compute gradient for quadratic problems"""
    
    return A @ x + b

def quadratic_Hess(A: Array):

    """ Compute Hessian for quadratic problems"""

    return A
    

