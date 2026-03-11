# All problems + associated gradients and Hessians
import numpy as np

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

