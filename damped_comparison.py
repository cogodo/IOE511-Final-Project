import numpy as np
import matplotlib.pyplot as plt

from benchmark import run_one, AlgoSpec
from objectives.base import SolverObjective
from runner import build_methods, build_problems

plt.ion()

"""
vardim.py
---------
Objective function, gradient, and Hessian for the Variable-Dimensionality
(vardim) benchmark function:

    f(x) = t^2 + t^4,   t = sum_{i=1}^{n} i * (x_i - 1)

Gradient:
    df/dx_j = (2t + 4t^3) * j

Hessian:
    d^2f / (dx_j dx_k) = (2 + 12t^2) * j * k
"""

import numpy as np


# ---------------------------------------------------------------------- #
# Internal helper                                                         #
# ---------------------------------------------------------------------- #

def _weights_and_t(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the index-weight vector i = [1, ..., n] and t = i · (x - 1)."""
    i = np.arange(1, x.size + 1, dtype=float)
    t = np.dot(i, x - 1.0)
    return i, t


# ---------------------------------------------------------------------- #
# Public interface                                                        #
# ---------------------------------------------------------------------- #

def vardim_f(x: np.ndarray) -> float:
    """
    Objective value of the vardim function.

        f(x) = t^2 + t^4,   t = sum_{i=1}^{n} i*(x_i - 1)

    Parameters
    ----------
    x : array_like, shape (n,)

    Returns
    -------
    f : float
    """
    _, t = _weights_and_t(np.asarray(x, dtype=float))
    return t**2 + t**4


def vardim_g(x: np.ndarray) -> np.ndarray:
    """
    Gradient of the vardim function.

        df/dx_j = (2t + 4t^3) * j

    Parameters
    ----------
    x : array_like, shape (n,)

    Returns
    -------
    g : np.ndarray, shape (n,)
    """
    i, t = _weights_and_t(np.asarray(x, dtype=float))
    g_val = (2.0 * t + 4.0 * t**3) * i
    return g_val.reshape(-1, 1)


def vardim_H(x: np.ndarray) -> np.ndarray:
    """
    Hessian of the vardim function.

        H_jk = (2 + 12t^2) * j * k

    Parameters
    ----------
    x : array_like, shape (n,)

    Returns
    -------
    H : np.ndarray, shape (n, n)
    """
    i, t = _weights_and_t(np.asarray(x, dtype=float))
    return (2.0 + 12.0 * t**2) * np.outer(i, i)

def vardim_x0(n: int) -> np.ndarray:
    """
    Standard starting point from Moré, Garbow & Hillstrom (1981).

        x_i = 1 - i/n,   i = 1, ..., n

    Parameters
    ----------
    n : int  — problem dimension

    Returns
    -------
    x0 : np.ndarray, shape (n,)
    """
    i = np.arange(1, n + 1, dtype=float)
    return 1.0 - i / n

def main():
    problems = build_problems()
    methods = build_methods()

    line_search_name = 'Wolfe'


    results_by_algorithm = {}
    for method_name in ['L-BFGS', 'D-L-BFGS', 'C-L-BFGS', 'DD-L-BFGS']:

        results = []
        for n in [1, 10, 100]:
            x_star = np.ones(n)
            x0 = vardim_x0(n).reshape(-1, 1)
            problem = SolverObjective(name = 'vardim', value=vardim_f, grad=vardim_g, hess=vardim_H, x0 = x0)

            options = methods[method_name + '_' + line_search_name][1]
            run_result = run_one(problem, AlgoSpec(label=method_name + '_ROSENBROCK2', algo_name=method_name, options=options))

            results.append(run_result)
        results_by_algorithm[method_name] = results


    pass


if __name__ == "__main__":
    main()


