"""
Scaling experiment: compare L-BFGS variants on Dixon-Price across dimensions.
"""

import numpy as np
import matplotlib.pyplot as plt

from benchmark import run_one, AlgoSpec
from objectives.base import SolverObjective
from runner import build_methods

import numpy as np

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



# ---------------------------------------------------------------------- #
# Internal helper                                                         #
# ---------------------------------------------------------------------- #

def _weights_and_t(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the index-weight vector i = [1, ..., n] and t = i . (x - 1)."""
    i = np.arange(1, x.size + 1, dtype=float)
    t = np.dot(i, x - 1.0)
    return i, t


def vardim_f(x: np.ndarray) -> float:
    _, t = _weights_and_t(np.asarray(x, dtype=float))
    return t**2 + t**4


def vardim_g(x: np.ndarray) -> np.ndarray:
    i, t = _weights_and_t(np.asarray(x, dtype=float))
    g_val = (2.0 * t + 4.0 * t**3) * i
    return g_val.reshape(-1, 1)


def vardim_H(x: np.ndarray) -> np.ndarray:
    i, t = _weights_and_t(np.asarray(x, dtype=float))
    return (2.0 + 12.0 * t**2) * np.outer(i, i)


def vardim_x0(n: int) -> np.ndarray:
    i = np.arange(1, n + 1, dtype=float)
    return 1.0 - i / n


def _check_and_split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.size % 2 != 0:
        raise ValueError(f"Extended Rosenbrock requires even dimension; got n={x.size}.")
    a = x[0::2]
    b = x[1::2]
    u = b - a ** 2
    return a, b, u


def rosenbrock_f(x: np.ndarray) -> float:
    a, _, u = _check_and_split(x)
    return float(np.sum(100.0 * u ** 2 + (1.0 - a) ** 2))


def rosenbrock_g(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    a, _, u = _check_and_split(x)
    g = np.empty_like(x)
    g[0::2] = -400.0 * u * a - 2.0 * (1.0 - a)
    g[1::2] = 200.0 * u
    return g.reshape(-1, 1)


def rosenbrock_H(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(np.squeeze(np.asarray(x, dtype=float)))
    n = x.size
    a, _, u = _check_and_split(x)
    H = np.zeros((n, n))
    for i, (ai, ui) in enumerate(zip(a, u)):
        r = 2 * i
        H[r, r] = -400.0 * ui + 800.0 * ai ** 2 + 2.0
        H[r, r + 1] = -400.0 * ai
        H[r + 1, r] = -400.0 * ai
        H[r + 1, r + 1] = 200.0
    return H


def rosenbrock_x0(n: int) -> np.ndarray:
    if n % 2 != 0:
        raise ValueError(f"n must be even; got n={n}.")
    x0 = np.ones(n)
    x0[0::2] = -1.2
    return x0


def _residuals(x: np.ndarray) -> np.ndarray:
    return 2.0 * x[1:] ** 2 - x[:-1]


def dixon_price_f(x: np.ndarray) -> float:
    x = np.atleast_1d(np.squeeze(np.asarray(x, dtype=float)))
    i = np.arange(2, x.size + 1, dtype=float)
    r = _residuals(x)
    return float((x[0] - 1.0) ** 2 + np.dot(i, r ** 2))


def dixon_price_g(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(np.squeeze(np.asarray(x, dtype=float)))
    n = x.size
    r = _residuals(x)
    i = np.arange(2, n + 1, dtype=float)

    g = np.empty(n)
    g[0] = 2.0 * (x[0] - 1.0) - 4.0 * r[0]
    if n > 2:
        g[1:-1] = 8.0 * i[:-1] * x[1:-1] * r[:-1] - 2.0 * i[1:] * r[1:]
    g[-1] = 8.0 * i[-1] * x[-1] * r[-1]

    return g.reshape(-1, 1)


def dixon_price_H(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(np.squeeze(np.asarray(x, dtype=float)))
    n = x.size
    r = _residuals(x)
    i = np.arange(2, n + 1, dtype=float)

    H = np.zeros((n, n))
    H[0, 0] = 6.0

    if n > 1:
        if n > 2:
            H[1:-1, 1:-1] = np.diag(
                8.0 * i[:-1] * r[:-1] + 32.0 * i[:-1] * x[1:-1] ** 2 + 2.0 * i[1:]
            )
        H[-1, -1] = 8.0 * i[-1] * r[-1] + 32.0 * i[-1] * x[-1] ** 2

        off = -8.0 * i * x[1:]
        H[np.arange(n - 1), np.arange(1, n)] = off
        H[np.arange(1, n), np.arange(n - 1)] = off

    return H


def dixon_price_x0(n: int) -> np.ndarray:
    return np.ones(n)


def main():
    methods = build_methods()
    line_search_name = 'Wolfe'

    dimensions = np.logspace(2, 3, 20).round().astype(int)
    dimensions_even = [d if d % 2 == 0 else d + 1 for d in dimensions]

    results_by_algorithm: dict[str, list] = {}
    for method_name in ['L-BFGS', 'D-L-BFGS', 'C-L-BFGS', 'DD-L-BFGS']:
        print(method_name)
        results = []
        for n in dimensions_even:
            print(n)
            x0 = dixon_price_x0(n).reshape(-1, 1)
            problem = SolverObjective(
                name='dixon_price',
                value=dixon_price_f,
                grad=dixon_price_g,
                hess=dixon_price_H,
                x0=x0,
            )

            options = methods[method_name + '_' + line_search_name][1]
            run_result = run_one(
                problem,
                AlgoSpec(label=f'{method_name}_DIXON_PRICE', algo_name=method_name, options=options),
            )
            results.append(run_result)
        results_by_algorithm[method_name] = results

    plt.figure()
    for method, results in results_by_algorithm.items():
        iterations = [r.iterations for r in results]
        plt.semilogx(dimensions_even, iterations, label=method, marker='o')

    plt.xlabel('Dimension')
    plt.ylabel('Iteration Count')
    plt.legend()
    plt.savefig('plots/dixon_price_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plots/dixon_price_scaling.png")


if __name__ == "__main__":
    main()
