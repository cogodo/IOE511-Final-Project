"""
Single benchmark script that runs all algorithm x problem experiments and
produces the "Summary of Results" tables (iterations, f-evals, g-evals,
CPU seconds) as both JSON and LaTeX.
"""

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from algorithms.base import SolverAlgorithm
from objectives.base import SolverObjective
from optSolver import optSolver
from options.base import LineSearchOptions, SolverOptions


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKTRACKING = LineSearchOptions(method="Backtracking")
WOLFE = LineSearchOptions(method="Wolfe")
CONSTANT = LineSearchOptions(method="Constant")


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------
def build_problems() -> list[SolverObjective]:
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

    return [
        SolverObjective(name="quad_10_10",      x0=q10.copy(),           f_star=None),
        SolverObjective(name="quad_10_1000",     x0=q10.copy(),           f_star=None),
        SolverObjective(name="quad_1000_10",     x0=q1000.copy(),         f_star=None),
        SolverObjective(name="quad_1000_1000",   x0=q1000.copy(),         f_star=None),
        SolverObjective(name="quartic_1",        x0=quartic_x0.copy(),    f_star=0.0),
        SolverObjective(name="quartic_2",        x0=quartic_x0.copy(),    f_star=0.0),
        SolverObjective(name="Rosenbrock-2",     x0=np.array([[-1.2], [1.0]]), f_star=0.0),
        SolverObjective(name="Rosenbrock-100",   x0=rosen_100_x0,         f_star=0.0),
        SolverObjective(name="datafit_2",        x0=np.ones((2, 1)),      f_star=None),
        SolverObjective(name="exp_10",           x0=exp_10_x0,            f_star=None),
        SolverObjective(name="exp_1000",         x0=exp_1000_x0,          f_star=None),
        SolverObjective(name="genhumps_5",       x0=genhumps_5_x0,        f_star=0.0),
    ]


# ---------------------------------------------------------------------------
# Algorithm + options definitions
# ---------------------------------------------------------------------------

@dataclass
class AlgoSpec:
    label: str
    algo_name: str
    options: SolverOptions


def build_algorithms() -> list[AlgoSpec]:
    return [
        AlgoSpec("GD_Backtracking",    "GradientDescent", SolverOptions(line_search=BACKTRACKING)),
        AlgoSpec("GD_Wolfe",           "GradientDescent", SolverOptions(line_search=WOLFE)),
        AlgoSpec("Newton_Backtracking","Newton",          SolverOptions(line_search=BACKTRACKING)),
        AlgoSpec("Newton_Wolfe",       "Newton",          SolverOptions(line_search=WOLFE)),
        AlgoSpec("TR-Newton-CG",       "TR-Newton-CG",    SolverOptions()),
        AlgoSpec("TR-SR1-CG",          "TR-SR1-CG",       SolverOptions()),
        AlgoSpec("BFGS_Backtracking",  "BFGS",            SolverOptions(line_search=BACKTRACKING)),
        AlgoSpec("BFGS_Wolfe",         "BFGS",            SolverOptions(line_search=WOLFE)),
        AlgoSpec("L-BFGS_Backtracking","L-BFGS",          SolverOptions(line_search=BACKTRACKING)),
        AlgoSpec("L-BFGS_Wolfe",       "L-BFGS",          SolverOptions(line_search=WOLFE)),
        AlgoSpec("DFP_Backtracking",   "DFP",             SolverOptions(line_search=BACKTRACKING)),
        AlgoSpec("DFP_Wolfe",          "DFP",             SolverOptions(line_search=WOLFE)),
    ]


# ---------------------------------------------------------------------------
# Single run wrapper
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    problem: str
    algorithm: str
    iterations: int
    nfev: int
    ngev: int
    cpu_time_s: float
    f_final: float
    grad_norm_final: float
    converged: bool
    error: str | None


def run_one(problem: SolverObjective, spec: AlgoSpec) -> RunResult:
    prob = copy.deepcopy(problem)
    method = SolverAlgorithm(name=spec.algo_name)
    f_vals: list[float] = []
    counters: dict = {}

    t0 = time.process_time()
    try:
        x, f = optSolver(prob, method, spec.options, f_vals=f_vals, counters=counters)
        elapsed = time.process_time() - t0
        nfev = counters.get('nfev', 0)
        ngev = counters.get('ngev', 0)
        g = prob.grad(x)
        gnorm = float(np.linalg.norm(g, ord=np.inf))
        g0 = prob.grad(prob.x0)
        gnorm0 = float(np.linalg.norm(g0, ord=np.inf))
        converged = gnorm <= spec.options.term_tol * max(gnorm0, 1.0)
        return RunResult(
            problem=problem.name,
            algorithm=spec.label,
            iterations=len(f_vals),
            nfev=nfev,
            ngev=ngev,
            cpu_time_s=round(elapsed, 4),
            f_final=float(np.squeeze(f)),
            grad_norm_final=gnorm,
            converged=converged,
            error=None,
        )
    except Exception as exc:
        elapsed = time.process_time() - t0
        return RunResult(
            problem=problem.name,
            algorithm=spec.label,
            iterations=len(f_vals),
            nfev=counters.get('nfev', 0),
            ngev=counters.get('ngev', 0),
            cpu_time_s=round(elapsed, 4),
            f_final=float("nan"),
            grad_norm_final=float("nan"),
            converged=False,
            error=str(exc)[:120],
        )


# ---------------------------------------------------------------------------
# Short display names (for tables)
# ---------------------------------------------------------------------------

SHORT_ALGO = {
    "GD_Backtracking": "GD-B", "GD_Wolfe": "GD-W",
    "Newton_Backtracking": "N-B", "Newton_Wolfe": "N-W",
    "TR-Newton-CG": "TR-N", "TR-SR1-CG": "TR-S",
    "BFGS_Backtracking": "B-B", "BFGS_Wolfe": "B-W",
    "L-BFGS_Backtracking": "LB-B", "L-BFGS_Wolfe": "LB-W", 
    "DFP_Backtracking": "DF-B", "DFP_Wolfe": "DF-W",
}

SHORT_PROB = {
    "quad_10_10": "Q-10/10", "quad_10_1000": "Q-10/1k",
    "quad_1000_10": "Q-1k/10", "quad_1000_1000": "Q-1k/1k",
    "quartic_1": "Quartic-1", "quartic_2": "Quartic-2",
    "Rosenbrock-2": "Rosen-2", "Rosenbrock-100": "Rosen-100",
    "datafit_2": "DataFit", "exp_10": "Exp-10",
    "exp_1000": "Exp-1000", "genhumps_5": "GenHumps",
}


# ---------------------------------------------------------------------------
# Console printing helpers
# ---------------------------------------------------------------------------

def print_problem_table(problem_name: str, results: list[RunResult]) -> None:
    hdr = (f"{'Algorithm':<24} {'Iters':>6} {'nfev':>6} {'ngev':>6} "
           f"{'CPU (s)':>10} {'f_final':>14} {'||g||_inf':>12} {'Status':<10}")
    sep = "-" * len(hdr)
    print(f"\n{'=' * len(hdr)}")
    print(f"  Problem: {problem_name}")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(sep)
    for r in results:
        if r.error:
            status = "ERROR"
        elif r.converged:
            status = "CONVERGED"
        else:
            status = "MAX_ITER"
        print(
            f"{r.algorithm:<24} {r.iterations:>6} {r.nfev:>6} {r.ngev:>6} "
            f"{r.cpu_time_s:>10.4f} {r.f_final:>14.6e} {r.grad_norm_final:>12.4e} {status:<10}"
            + (f"  {r.error}" if r.error else "")
        )
    print()


def print_global_summary(all_results: list[RunResult], algo_specs: list[AlgoSpec]) -> None:
    algo_labels = [s.label for s in algo_specs]
    problems = list(dict.fromkeys(r.problem for r in all_results))

    print("\n" + "=" * 80)
    print("  GLOBAL SUMMARY: convergence matrix  (C=converged, M=max_iter, E=error)")
    print("=" * 80)

    col_w = 8
    header = f"{'Problem':<24}" + "".join(f"{a[:col_w]:>{col_w}}" for a in algo_labels)
    print(header)
    print("-" * len(header))

    for pname in problems:
        row = f"{pname:<24}"
        for alabel in algo_labels:
            r = next((r for r in all_results if r.problem == pname and r.algorithm == alabel), None)
            if r is None:
                cell = "-"
            elif r.error:
                cell = "E"
            elif r.converged:
                cell = "C"
            else:
                cell = "M"
            row += f"{cell:>{col_w}}"
        print(row)

    print()
    total = len(all_results)
    n_conv = sum(1 for r in all_results if r.converged)
    n_err = sum(1 for r in all_results if r.error)
    n_max = total - n_conv - n_err
    print(f"Total runs: {total}  |  Converged: {n_conv}  |  Max-iter: {n_max}  |  Errors: {n_err}")
    print()


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def make_latex_table(
    metric: str,
    caption: str,
    label: str,
    lookup: dict[tuple[str, str], dict],
    prob_names: list[str],
    algo_names: list[str],
    fmt: str = "d",
) -> str:
    n = len(algo_names)
    col_spec = "l" + "r" * n
    header_cells = " & ".join(f"\\romark{{{SHORT_ALGO[a]}}}" for a in algo_names)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{caption}  A cell marked F indicates the algorithm failed (hit max iterations) on that problem.}}",
        f"\\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\resizebox{\textwidth}{!}{",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"Problem & {header_cells} \\\\",
        r"\midrule",
    ]

    for p in prob_names:
        cells = [SHORT_PROB[p]]
        for a in algo_names:
            r = lookup[(p, a)]
            if not r["converged"]:
                cells.append("F")
            else:
                val = r[metric]
                if fmt == "d":
                    cells.append(str(int(val)))
                else:
                    cells.append(f"{val:.4f}")
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}"]
    return "\n".join(lines)


def write_latex_tables(all_results: list[RunResult], out_path: Path) -> None:
    serializable = _serialize_results(all_results)
    prob_names = list(dict.fromkeys(r["problem"] for r in serializable))
    algo_names = list(dict.fromkeys(r["algorithm"] for r in serializable))
    lookup = {(r["problem"], r["algorithm"]): r for r in serializable}

    preamble = r"""\newcommand{\romark}[1]{\rotatebox{70}{\makebox[0pt][l]{#1}}}"""
    tables = [
        make_latex_table("iterations", "Summary of Results: Number of Iterations.", "tab:iters", lookup, prob_names, algo_names),
        make_latex_table("nfev", "Summary of Results: Number of Function Evaluations.", "tab:nfev", lookup, prob_names, algo_names),
        make_latex_table("ngev", "Summary of Results: Number of Gradient Evaluations.", "tab:ngev", lookup, prob_names, algo_names),
        make_latex_table("cpu_time_s", "Summary of Results: CPU Seconds.", "tab:cpu", lookup, prob_names, algo_names, fmt="f"),
    ]

    output = preamble + "\n\n" + "\n\n".join(tables)
    out_path.write_text(output)
    print(f"LaTeX tables written to {out_path} ({len(output)} chars)")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _serialize_results(all_results: list[RunResult]) -> list[dict]:
    return [
        {
            "problem": r.problem,
            "algorithm": r.algorithm,
            "iterations": r.iterations,
            "nfev": r.nfev,
            "ngev": r.ngev,
            "cpu_time_s": r.cpu_time_s,
            "f_final": r.f_final if not np.isnan(r.f_final) else None,
            "grad_norm_final": r.grad_norm_final if not np.isnan(r.grad_norm_final) else None,
            "converged": r.converged,
            "error": r.error,
        }
        for r in all_results
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    problems = build_problems()
    algo_specs = build_algorithms()
    all_results: list[RunResult] = []

    total = len(problems) * len(algo_specs)
    run_idx = 0

    for prob in problems:
        prob_results: list[RunResult] = []
        for spec in algo_specs:
            run_idx += 1
            tag = f"[{run_idx}/{total}]"
            print(f"{tag} {prob.name:<24} x {spec.label:<24} ... ", end="", flush=True)

            result = run_one(prob, spec)
            prob_results.append(result)

            if result.error:
                print(f"ERROR ({result.cpu_time_s:.2f}s): {result.error}")
            elif result.converged:
                print(f"CONVERGED  iters={result.iterations:<5} nfev={result.nfev:<5} ngev={result.ngev:<5} f={result.f_final:.6e}  t={result.cpu_time_s:.2f}s")
            else:
                print(f"MAX_ITER   iters={result.iterations:<5} nfev={result.nfev:<5} ngev={result.ngev:<5} f={result.f_final:.6e}  t={result.cpu_time_s:.2f}s")

        print_problem_table(prob.name, prob_results)
        all_results.extend(prob_results)

    print_global_summary(all_results, algo_specs)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(_serialize_results(all_results), indent=2))
    print(f"Results saved to {json_path}")

    tex_path = out_dir / "summary_tables.tex"
    write_latex_tables(all_results, tex_path)


if __name__ == "__main__":
    main()
