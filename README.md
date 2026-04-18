# IOE 511 Final Project

Numerical comparison of optimization algorithms, with a focus on BFGS variants (damped, cautious, limited-memory, and combinations).

## Structure

```
optSolver.py           Main solver loop (dispatches problem + algorithm)
algorithms/            Step functions for each algorithm
  algorithms.py        GD, Newton, TR, BFGS, D-BFGS, C-BFGS, DD-BFGS, L-BFGS, DFP, etc.
  utils.py             Line search (backtracking, weak Wolfe), CG, trust-region helpers
  base.py              SolverAlgorithm dataclass
objectives/            Test problem definitions (quadratics, Rosenbrock, quartics, etc.)
  functions.py         Objective, gradient, Hessian for each problem
  base.py              SolverObjective dataclass
  data/                Precomputed Q matrices (.mat) for quadratic problems
options/               Solver configuration dataclasses
benchmark.py           Full benchmark: all algorithms x all problems -> JSON
runner.py              Ad-hoc experiments and convergence plots
gridsearch.py          Hyperparameter grid search -> Excel
damped_comparison.py   Scaling experiment: L-BFGS variants on Dixon-Price
make_plots.py          Generate all comparison plots from benchmark JSON
make_tables.py         Generate LaTeX summary tables from benchmark JSON
data/                  Grid search result spreadsheets
plots/                 Generated figures (performance profiles, heatmaps, bar charts)
```

## Setup

Requires Python >= 3.13.

```bash
uv sync
```

Or with pip:

```bash
pip install numpy scipy einops matplotlib scikit-learn pandas tqdm
```

## Reproducing Results

```bash
# Run full benchmark (writes results/benchmark_results.json)
python benchmark.py

# Generate plots (reads benchmark JSON, writes to plots/)
python make_plots.py

# Generate LaTeX tables (writes results/summary_tables.tex)
python make_tables.py
```

## Algorithms

| Family | Variants |
|--------|----------|
| Gradient Descent | Constant step, Backtracking, Wolfe |
| Newton | Backtracking, Wolfe (with modified Cholesky) |
| Trust Region | TR-Newton-CG, TR-SR1-CG |
| BFGS | Standard, Damped (D-BFGS), Cautious (C-BFGS), Double-Damped (DD-BFGS) |
| L-BFGS | Standard, Damped (D-L-BFGS), Cautious (C-L-BFGS), Double-Damped (DD-L-BFGS) |
| DFP | Backtracking, Wolfe |

## Test Problems

Quadratics (10/1000 dim, condition 10/1000), Quartic 1 & 2, Rosenbrock (2 & 100 dim), Data Fitting, Exponential (10 & 1000 dim), Generalized Humps.
