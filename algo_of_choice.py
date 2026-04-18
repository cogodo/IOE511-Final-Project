"""Run L-BFGS with Wolfe line search on both Rosenbrock problems."""

import copy

import numpy as np

from algorithms.base import SolverAlgorithm
from objectives.base import SolverObjective
from optSolver import optSolver
from options.base import LineSearchOptions, SolverOptions


WOLFE = LineSearchOptions(method="Wolfe")
OPTIONS = SolverOptions(line_search=WOLFE)
ALGO_NAME = "L-BFGS"


def build_rosenbrock_problems() -> list[SolverObjective]:
    rosen_100_x0 = np.ones((100, 1))
    rosen_100_x0[0] = -1.2

    return [
        SolverObjective(name="Rosenbrock-2", x0=np.array([[-1.2], [1.0]]), f_star=0.0),
        SolverObjective(name="Rosenbrock-100", x0=rosen_100_x0, f_star=0.0),
    ]


def main() -> None:
    for problem in build_rosenbrock_problems():
        prob = copy.deepcopy(problem)
        f_vals: list[float] = []
        counters: dict = {}

        print(f"--- {prob.name} ---")
        x, f = optSolver(prob, SolverAlgorithm(ALGO_NAME), OPTIONS, f_vals=f_vals, counters=counters)

        print(f"  iterations : {len(f_vals)}")
        print(f"  f(x*)      : {f}")
        print(f"  ||x*||     : {np.linalg.norm(x)}")
        print(f"  f evals    : {counters['nfev']}")
        print(f"  g evals    : {counters['ngev']}")
        print()


if __name__ == "__main__":
    main()
