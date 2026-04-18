"""
Runner for ad-hoc experiments and convergence plots.

Usage:
    python runner.py                        # runs default experiment
    python runner.py Rosenbrock-2           # plots all methods on a specific problem
    python runner.py quad_10_10 BFGS_Wolfe  # runs a single (problem, method) pair
"""

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithms.base import SolverAlgorithm
from objectives.base import SolverObjective
from optSolver import optSolver
from options.base import (
    LineSearchOptions,
    SolverOptions,
)


# ---------------------------------------------------------------------------
# Problem catalogue
# ---------------------------------------------------------------------------
def build_problems() -> dict[str, SolverObjective]:
    np.random.seed(0)
    q10 = 20 * np.random.normal(size=(10, 1)) - 10
    np.random.seed(0)
    q1000 = 20 * np.random.normal(size=(1000, 1)) - 10

    quartic_x0 = np.array([[np.cos(70)], [np.sin(70)], [np.cos(70)], [np.sin(70)]])

    rosen_100_x0 = np.ones((100, 1))
    rosen_100_x0[0] = -1.2

    exp_10_x0 = np.zeros((10, 1))
    exp_10_x0[0] = 1.0

    exp_1000_x0 = np.zeros((1000, 1))
    exp_1000_x0[0] = 1.0

    genhumps_5_x0 = 506.2 * np.ones((5, 1))
    genhumps_5_x0[0] = -506.2

    return {
        "quad_10_10":      SolverObjective(name="quad_10_10",      x0=q10.copy(),           f_star=None),
        "quad_10_1000":    SolverObjective(name="quad_10_1000",    x0=q10.copy(),           f_star=None),
        "quad_1000_10":    SolverObjective(name="quad_1000_10",    x0=q1000.copy(),         f_star=None),
        "quad_1000_1000":  SolverObjective(name="quad_1000_1000",  x0=q1000.copy(),         f_star=None),
        "quartic_1":       SolverObjective(name="quartic_1",       x0=quartic_x0.copy(),    f_star=0.0),
        "quartic_2":       SolverObjective(name="quartic_2",       x0=quartic_x0.copy(),    f_star=0.0),
        "Rosenbrock-2":    SolverObjective(name="Rosenbrock-2",    x0=np.array([[-1.2], [1.0]]), f_star=0.0),
        "Rosenbrock-100":  SolverObjective(name="Rosenbrock-100",  x0=rosen_100_x0,         f_star=0.0),
        "datafit_2":       SolverObjective(name="datafit_2",       x0=np.ones((2, 1)),      f_star=None),
        "exp_10":          SolverObjective(name="exp_10",          x0=exp_10_x0,            f_star=None),
        "exp_1000":        SolverObjective(name="exp_1000",        x0=exp_1000_x0,          f_star=None),
        "genhumps_5":      SolverObjective(name="genhumps_5",      x0=genhumps_5_x0,        f_star=0.0),
    }


# ---------------------------------------------------------------------------
# Method catalogue  (label -> (algorithm_name, SolverOptions))
# ---------------------------------------------------------------------------
BACKTRACKING = LineSearchOptions(method="Backtracking")
WOLFE = LineSearchOptions(method="Wolfe")
CONSTANT = LineSearchOptions(method="Constant")


def build_methods() -> dict[str, tuple[str, SolverOptions]]:
    return {
        "GD_Constant":        ("GradientDescent", SolverOptions(line_search=CONSTANT)),
        "GD_Backtracking":    ("GradientDescent", SolverOptions(line_search=BACKTRACKING)),
        "GD_Wolfe":           ("GradientDescent", SolverOptions(line_search=WOLFE)),
        "Newton_Backtracking":("Newton",          SolverOptions(line_search=BACKTRACKING)),
        "Newton_Wolfe":       ("Newton",          SolverOptions(line_search=WOLFE)),
        "TR-Newton-CG":       ("TR-Newton-CG",    SolverOptions()),
        "TR-SR1-CG":          ("TR-SR1-CG",       SolverOptions()),
        "BFGS_Backtracking":  ("BFGS",            SolverOptions(line_search=BACKTRACKING)),
        "BFGS_Wolfe":         ("BFGS",            SolverOptions(line_search=WOLFE)),
        "D-BFGS_Wolfe":       ("D-BFGS",          SolverOptions(line_search=WOLFE)),
        "C-BFGS_Wolfe":       ("C-BFGS",          SolverOptions(line_search=WOLFE)),
        "DD-BFGS_Wolfe":      ("DD-BFGS",         SolverOptions(line_search=WOLFE)),
        "L-BFGS_Wolfe":       ("L-BFGS",          SolverOptions(line_search=WOLFE)),
        "D-L-BFGS_Wolfe":     ("D-L-BFGS",        SolverOptions(line_search=WOLFE)),
        "C-L-BFGS_Wolfe":     ("C-L-BFGS",        SolverOptions(line_search=WOLFE)),
        "DD-L-BFGS_Wolfe":    ("DD-L-BFGS",       SolverOptions(line_search=WOLFE)),
        "DFP_Backtracking":   ("DFP",             SolverOptions(line_search=BACKTRACKING)),
        "DFP_Wolfe":          ("DFP",             SolverOptions(line_search=WOLFE)),
    }


# ---------------------------------------------------------------------------
# Color palette grouped by algorithm family
# ---------------------------------------------------------------------------
FAMILY_CMAPS: list[tuple[str, list[str]]] = [
    ("Blues",   ["GD_Constant", "GD_Backtracking", "GD_Wolfe"]),
    ("Greens",  ["Newton_Backtracking", "Newton_Wolfe"]),
    ("Oranges", ["TR-Newton-CG", "TR-SR1-CG"]),
    ("RdPu",    ["BFGS_Backtracking", "BFGS_Wolfe"]),
    ("YlOrBr",  ["D-BFGS_Wolfe", "DD-BFGS_Wolfe", "C-BFGS_Wolfe"]),
    ("cividis",    ["L-BFGS_Wolfe", "D-L-BFGS_Wolfe"]),
    ("Purples", ["DFP_Backtracking", "DFP_Wolfe"]),
]


def _build_color_map() -> dict[str, tuple]:
    colors: dict[str, tuple] = {}
    for cmap_name, labels in FAMILY_CMAPS:
        cmap = plt.get_cmap(cmap_name)
        n = len(labels)
        for i, label in enumerate(labels):
            colors[label] = cmap(0.4 + 0.3 * i / max(n - 1, 1))
    return colors


# ---------------------------------------------------------------------------
# Convergence plot: f - f* vs iterations for one problem, all methods
# ---------------------------------------------------------------------------
def plot_convergence(
    problem: SolverObjective,
    methods: dict[str, tuple[str, SolverOptions]],
    title: str | None = None,
    save_dir: str = "plots",
) -> None:
    if problem.f_star is None:
        print(f"Skipping plot for {problem.name}: f_star not set")
        return

    colors = _build_color_map()
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, (algo_name, opts) in methods.items():
        prob = copy.deepcopy(problem)
        f_vals: list[float] = []
        try:
            optSolver(prob, SolverAlgorithm(algo_name), opts, f_vals=f_vals)
        except Exception as exc:
            print(f"  {label}: ERROR -- {exc}")
            continue

        if not f_vals:
            continue

        residuals = np.array(f_vals) - problem.f_star
        residuals = np.maximum(residuals, 1e-16)
        ax.plot(residuals, color=colors.get(label), label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Iteration k")
    ax.set_ylabel("f(x_k) - f*")
    ax.set_title(title or problem.name)
    ax.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center", ncol=5, fontsize=7)
    fig.tight_layout()

    out = Path(save_dir)
    out.mkdir(exist_ok=True)
    fname = out / f"{title or problem.name}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Single run helper
# ---------------------------------------------------------------------------
def run_single(
    problem: SolverObjective,
    algo_name: str,
    options: SolverOptions,
) -> None:
    prob = copy.deepcopy(problem)
    x, f = optSolver(prob, SolverAlgorithm(algo_name), options)
    print(f"  x = {x.flatten()}")
    print(f"  f = {f}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    problems = build_problems()
    methods = build_methods()

    args = sys.argv[1:]

    if len(args) == 0:
        prob = problems["Rosenbrock-2"]
        print(f"Running convergence plot for {prob.name} with all methods...")
        plot_convergence(prob, methods, title=prob.name)
        return

    if len(args) == 1:
        pname = args[0]
        if pname not in problems:
            print(f"Unknown problem '{pname}'. Available: {', '.join(problems)}")
            sys.exit(1)
        prob = problems[pname]
        print(f"Running convergence plot for {prob.name} with all methods...")
        plot_convergence(prob, methods, title=prob.name)
        return

    if len(args) == 2:
        pname, mlabel = args
        if pname not in problems:
            print(f"Unknown problem '{pname}'. Available: {', '.join(problems)}")
            sys.exit(1)
        if mlabel not in methods:
            print(f"Unknown method '{mlabel}'. Available: {', '.join(methods)}")
            sys.exit(1)
        prob = problems[pname]
        algo_name, opts = methods[mlabel]
        print(f"{pname} x {mlabel}:")
        run_single(prob, algo_name, opts)
        return

    print(__doc__)
    sys.exit(1)


if __name__ == "__main__":
    main()
