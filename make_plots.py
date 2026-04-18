"""
Generate comparison plots from benchmark_results.json.

Produces:
  - Dolan-Moré performance profiles (iterations, nfev, ngev, cpu_time)
  - Per-family performance profiles (GD, Newton, TR, BFGS, L-BFGS, DFP)
  - Grouped bar charts per problem
  - Convergence heatmap
  - Iteration vs CPU scatter
  - BFGS variant comparison
  - Line search comparison
  - QN vs second-order comparison
"""

import json
from pathlib import Path
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ── Load data ────────────────────────────────────────────────────────────────

DATA = json.loads(Path("results/benchmark_results.json").read_text())
OUTDIR = Path("plots")
OUTDIR.mkdir(exist_ok=True)

DIR_PERFPROF = OUTDIR / "performance_profiles"
DIR_PERPROBLEM = OUTDIR / "per_problem_bars"
DIR_BFGS = OUTDIR / "bfgs_variants"
DIR_EVAL = OUTDIR / "eval_breakdowns"
DIR_HEAT = OUTDIR / "heatmaps"
DIR_SUMMARY = OUTDIR / "summaries"
DIR_LS = OUTDIR / "linesearch"
for _d in [DIR_PERFPROF, DIR_PERPROBLEM, DIR_BFGS, DIR_EVAL, DIR_HEAT, DIR_SUMMARY, DIR_LS]:
    _d.mkdir(exist_ok=True)

PROBLEMS = list(dict.fromkeys(r["problem"] for r in DATA))
ALGORITHMS = list(dict.fromkeys(r["algorithm"] for r in DATA))
LOOKUP = {(r["problem"], r["algorithm"]): r for r in DATA}

# ── Algorithm families ───────────────────────────────────────────────────────

FAMILIES = {
    "Gradient Descent": ["GD_Constant", "GD_Backtracking", "GD_Wolfe"],
    "Newton": ["Newton_Backtracking", "Newton_Wolfe"],
    "Trust Region": ["TR-Newton-CG", "TR-SR1-CG"],
    "BFGS": ["BFGS_Backtracking", "BFGS_Wolfe", "D-BFGS_Wolfe", "DD-BFGS_Wolfe", "C-BFGS_Wolfe"],
    "L-BFGS": ["L-BFGS_Wolfe", "D-L-BFGS_Wolfe"],
    "DFP": ["DFP_Backtracking", "DFP_Wolfe"],
}

# "Best-in-class" representatives for cross-family comparison
REPRESENTATIVES = {
    "GD (Backtracking)": "GD_Backtracking",
    "Newton (Wolfe)": "Newton_Wolfe",
    "TR-Newton-CG": "TR-Newton-CG",
    "TR-SR1-CG": "TR-SR1-CG",
    "BFGS (Wolfe)": "BFGS_Wolfe",
    "D-BFGS": "D-BFGS_Wolfe",
    "L-BFGS": "L-BFGS_Wolfe",
    "D-L-BFGS": "D-L-BFGS_Wolfe",
    "DFP (Backtracking)": "DFP_Backtracking",
}

SHORT = {
    "GD_Constant": "GD-C", "GD_Backtracking": "GD-B", "GD_Wolfe": "GD-W",
    "Newton_Backtracking": "N-B", "Newton_Wolfe": "N-W",
    "TR-Newton-CG": "TR-N", "TR-SR1-CG": "TR-S",
    "BFGS_Backtracking": "B-B", "BFGS_Wolfe": "B-W",
    "D-BFGS_Wolfe": "D-B", "DD-BFGS_Wolfe": "DD-B", "C-BFGS_Wolfe": "C-B",
    "L-BFGS_Wolfe": "L-B", "D-L-BFGS_Wolfe": "DL-B",
    "DFP_Backtracking": "DF-B", "DFP_Wolfe": "DF-W",
}

SHORT_PROB = {
    "quad_10_10": "Q-10/10", "quad_10_1000": "Q-10/1k",
    "quad_1000_10": "Q-1k/10", "quad_1000_1000": "Q-1k/1k",
    "quartic_1": "Quart-1", "quartic_2": "Quart-2",
    "Rosenbrock-2": "Rosen-2", "Rosenbrock-100": "Rosen-100",
    "datafit_2": "DataFit", "exp_10": "Exp-10",
    "exp_1000": "Exp-1k", "genhumps_5": "GenHumps",
}

# ── Color palette ────────────────────────────────────────────────────────────

FAMILY_COLORS = {
    "Gradient Descent": "#3b82f6",
    "Newton": "#22c55e",
    "Trust Region": "#f97316",
    "BFGS": "#ef4444",
    "L-BFGS": "#8b5cf6",
    "DFP": "#ec4899",
}

ALGO_COLORS = {}
for fam, members in FAMILIES.items():
    base = mcolors.to_rgb(FAMILY_COLORS[fam])
    for i, algo in enumerate(members):
        # lighten/darken within the family
        factor = 0.6 + 0.4 * i / max(len(members) - 1, 1)
        ALGO_COLORS[algo] = tuple(c * factor for c in base)


# ── Helpers ──────────────────────────────────────────────────────────────────

FAIL_PENALTY = 1e4  # multiplier for performance ratio when solver fails


def _perf_ratios(metric: str, algo_subset: list[str]) -> dict[str, np.ndarray]:
    """
    Compute Dolan-Moré performance ratios for each algorithm.
    Returns {algo_label: sorted_ratio_array}.
    """
    n_p = len(PROBLEMS)
    raw = {a: np.full(n_p, np.inf) for a in algo_subset}

    for ip, p in enumerate(PROBLEMS):
        vals = {}
        for a in algo_subset:
            r = LOOKUP.get((p, a))
            if r and r["converged"]:
                vals[a] = r[metric]
        if not vals:
            continue
        best = min(vals.values())
        if best <= 0:
            best = 1e-12
        for a in algo_subset:
            if a in vals:
                raw[a][ip] = vals[a] / best
            else:
                raw[a][ip] = FAIL_PENALTY

    return raw


def _plot_performance_profile(
    metric: str,
    algo_subset: list[str],
    labels: dict[str, str] | None = None,
    colors: dict[str, tuple] | None = None,
    title_suffix: str = "",
    fname_suffix: str = "",
    tau_max: float = 50.0,
):
    """Plot a Dolan-Moré performance profile."""
    labels = labels or SHORT
    colors = colors or ALGO_COLORS
    ratios = _perf_ratios(metric, algo_subset)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    taus = np.linspace(1, tau_max, 2000)

    styles = cycle(["-", "--", "-.", ":"])
    for algo in algo_subset:
        r = ratios[algo]
        fracs = np.array([np.mean(r <= t) for t in taus])
        lbl = labels.get(algo, algo)
        ax.plot(taus, fracs, label=lbl, color=colors.get(algo), linewidth=1.8, linestyle=next(styles))

    ax.set_xlabel(r"Performance ratio $\tau$", fontsize=12)
    ax.set_ylabel(r"Fraction of problems solved $\rho_s(\tau)$", fontsize=12)
    ax.set_title(f"Performance Profile — {metric}{title_suffix}", fontsize=13)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(1, tau_max)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIR_PERFPROF / f"perfprof_{metric}{fname_suffix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── 1. Global performance profiles (all algorithms) ─────────────────────────

def plot_global_performance_profiles():
    for metric in ["iterations", "nfev", "ngev", "cpu_time_s"]:
        _plot_performance_profile(metric, ALGORITHMS, title_suffix=" (all algorithms)", fname_suffix="_all")
    print("  [done] global performance profiles")


# ── 2. Per-family performance profiles ───────────────────────────────────────

def plot_family_performance_profiles():
    for fam, members in FAMILIES.items():
        if len(members) < 2:
            continue
        for metric in ["iterations", "cpu_time_s"]:
            _plot_performance_profile(
                metric, members,
                title_suffix=f" — {fam}",
                fname_suffix=f"_{fam.replace(' ', '_').lower()}",
                tau_max=30.0,
            )
    print("  [done] per-family performance profiles")


# ── 3. Cross-family performance profiles (representatives) ──────────────────

def plot_representative_profiles():
    algo_subset = list(REPRESENTATIVES.values())
    rep_labels = {v: k for k, v in REPRESENTATIVES.items()}
    rep_colors = {}
    for label, algo in REPRESENTATIVES.items():
        for fam, members in FAMILIES.items():
            if algo in members:
                rep_colors[algo] = FAMILY_COLORS[fam]
                break

    for metric in ["iterations", "nfev", "cpu_time_s"]:
        _plot_performance_profile(
            metric, algo_subset,
            labels=rep_labels,
            colors=rep_colors,
            title_suffix=" (best per family)",
            fname_suffix="_representatives",
            tau_max=40.0,
        )
    print("  [done] representative performance profiles")


# ── 4. Convergence heatmap ───────────────────────────────────────────────────

def plot_convergence_heatmap():
    n_a, n_p = len(ALGORITHMS), len(PROBLEMS)
    matrix = np.zeros((n_p, n_a))

    for ip, p in enumerate(PROBLEMS):
        for ia, a in enumerate(ALGORITHMS):
            r = LOOKUP.get((p, a))
            if r is None:
                matrix[ip, ia] = -1
            elif r["error"]:
                matrix[ip, ia] = -1
            elif r["converged"]:
                matrix[ip, ia] = 1
            else:
                matrix[ip, ia] = 0

    cmap = mcolors.ListedColormap(["#ef4444", "#fbbf24", "#22c55e"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n_a))
    ax.set_xticklabels([SHORT.get(a, a) for a in ALGORITHMS], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n_p))
    ax.set_yticklabels([SHORT_PROB.get(p, p) for p in PROBLEMS], fontsize=9)
    ax.set_title("Convergence Status", fontsize=13)

    legend_elements = [
        Patch(facecolor="#22c55e", label="Converged"),
        Patch(facecolor="#fbbf24", label="Max Iter"),
        Patch(facecolor="#ef4444", label="Error"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(DIR_HEAT / "convergence_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [done] convergence heatmap")


# ── 5. Iteration count heatmap (log scale) ──────────────────────────────────

def plot_iteration_heatmap():
    n_a, n_p = len(ALGORITHMS), len(PROBLEMS)
    matrix = np.full((n_p, n_a), np.nan)

    for ip, p in enumerate(PROBLEMS):
        for ia, a in enumerate(ALGORITHMS):
            r = LOOKUP.get((p, a))
            if r and r["converged"]:
                matrix[ip, ia] = np.log10(max(r["iterations"], 1))

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log₁₀(iterations)", fontsize=10)

    ax.set_xticks(range(n_a))
    ax.set_xticklabels([SHORT.get(a, a) for a in ALGORITHMS], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n_p))
    ax.set_yticklabels([SHORT_PROB.get(p, p) for p in PROBLEMS], fontsize=9)
    ax.set_title("Iteration Count (converged solvers only, log scale)", fontsize=13)

    # gray out non-converged
    for ip in range(n_p):
        for ia in range(n_a):
            if np.isnan(matrix[ip, ia]):
                ax.add_patch(plt.Rectangle((ia - 0.5, ip - 0.5), 1, 1, fill=True, color="#d1d5db", zorder=2))

    fig.tight_layout()
    fig.savefig(DIR_HEAT / "iteration_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [done] iteration heatmap")


# ── 6. Grouped bar charts per problem ───────────────────────────────────────

def plot_per_problem_bars():
    for problem in PROBLEMS:
        algos_conv = []
        iters_conv = []
        colors_conv = []
        for a in ALGORITHMS:
            r = LOOKUP.get((problem, a))
            if r and r["converged"]:
                algos_conv.append(SHORT.get(a, a))
                iters_conv.append(r["iterations"])
                colors_conv.append(ALGO_COLORS.get(a, "#999"))

        if not algos_conv:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(algos_conv) * 0.6), 5))
        x = np.arange(len(algos_conv))
        ax.bar(x, iters_conv, color=colors_conv, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(algos_conv, rotation=50, ha="right", fontsize=8)
        ax.set_ylabel("Iterations")
        ax.set_title(f"Iterations to Convergence — {SHORT_PROB.get(problem, problem)}", fontsize=12)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        safe = problem.replace("-", "_").replace(" ", "_")
        fig.savefig(DIR_PERPROBLEM / f"bar_iters_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("  [done] per-problem bar charts")


# ── 7. Scatter: iterations vs CPU time ──────────────────────────────────────

def plot_iter_vs_cpu():
    fig, ax = plt.subplots(figsize=(9, 6))

    for fam, members in FAMILIES.items():
        xs, ys = [], []
        for a in members:
            for p in PROBLEMS:
                r = LOOKUP.get((p, a))
                if r and r["converged"]:
                    xs.append(r["iterations"])
                    ys.append(r["cpu_time_s"])
        if xs:
            ax.scatter(xs, ys, label=fam, color=FAMILY_COLORS[fam], alpha=0.6, s=30, edgecolors="white", linewidth=0.4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iterations", fontsize=12)
    ax.set_ylabel("CPU Time (s)", fontsize=12)
    ax.set_title("Iterations vs CPU Time (converged runs, by family)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIR_SUMMARY / "scatter_iter_vs_cpu.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [done] iter vs cpu scatter")


# ── 8. Stacked bar: function evals = nfev + ngev ────────────────────────────

def plot_eval_breakdown():
    """Stacked bar of nfev/ngev for representative algorithms on each problem."""
    reps = REPRESENTATIVES
    rep_list = list(reps.keys())
    algo_list = [reps[k] for k in rep_list]

    for problem in PROBLEMS:
        nfevs, ngevs, labels, colors = [], [], [], []
        for label, algo in zip(rep_list, algo_list):
            r = LOOKUP.get((problem, algo))
            if r and r["converged"]:
                nfevs.append(r["nfev"])
                ngevs.append(r["ngev"])
                labels.append(label)
                for fam, members in FAMILIES.items():
                    if algo in members:
                        colors.append(FAMILY_COLORS[fam])
                        break

        if not labels:
            continue

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
        ax.bar(x, nfevs, label="f evals", color=colors, alpha=0.85, edgecolor="white")
        ax.bar(x, ngevs, bottom=nfevs, label="g evals", color=colors, alpha=0.45, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
        ax.set_ylabel("Number of Evaluations")
        ax.set_title(f"Function + Gradient Evaluations — {SHORT_PROB.get(problem, problem)}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        safe = problem.replace("-", "_").replace(" ", "_")
        fig.savefig(DIR_EVAL / f"eval_breakdown_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("  [done] eval breakdown charts")


# ── 9. Family-grouped bar: average iters/time across problems ───────────────

def plot_family_summary_bars():
    """Average iterations and CPU time per algorithm (only converged problems counted)."""
    for metric, ylabel, fname in [
        ("iterations", "Avg Iterations", "avg_iters"),
        ("cpu_time_s", "Avg CPU Time (s)", "avg_cpu"),
    ]:
        means, names, colors = [], [], []
        for a in ALGORITHMS:
            vals = [LOOKUP[(p, a)][metric] for p in PROBLEMS if LOOKUP[(p, a)]["converged"]]
            if not vals:
                continue
            means.append(np.mean(vals))
            names.append(SHORT.get(a, a))
            colors.append(ALGO_COLORS.get(a, "#999"))

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 5))
        x = np.arange(len(names))
        ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=50, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} (converged problems only)", fontsize=12)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(DIR_SUMMARY / f"{fname}_all.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("  [done] family summary bars")


# ── 10. Convergence rate bar chart ──────────────────────────────────────────

def plot_convergence_rate():
    """Fraction of problems each algorithm converged on."""
    rates, names, colors = [], [], []
    for a in ALGORITHMS:
        total = len(PROBLEMS)
        conv = sum(1 for p in PROBLEMS if LOOKUP[(p, a)]["converged"])
        rates.append(conv / total)
        names.append(SHORT.get(a, a))
        colors.append(ALGO_COLORS.get(a, "#999"))

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 5))
    x = np.arange(len(names))
    bars = ax.bar(x, rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=50, ha="right", fontsize=8)
    ax.set_ylabel("Convergence Rate")
    ax.set_title("Fraction of Problems Converged", fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.0%}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(DIR_SUMMARY / "convergence_rate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [done] convergence rate bar")


# ── 11. Per-family comparison: small-dim vs large-dim ────────────────────────

def plot_small_vs_large():
    """Compare algorithm iterations on small-dim problems vs large-dim problems."""
    small = ["quad_10_10", "quad_10_1000", "quartic_1", "quartic_2", "Rosenbrock-2", "datafit_2", "exp_10", "genhumps_5"]
    large = ["quad_1000_10", "quad_1000_1000", "Rosenbrock-100", "exp_1000"]

    for group_name, group_probs in [("small_dim", small), ("large_dim", large)]:
        reps = REPRESENTATIVES
        rep_list = list(reps.keys())
        algo_list = [reps[k] for k in rep_list]

        means, names, colors_list = [], [], []
        for label, algo in zip(rep_list, algo_list):
            vals = [LOOKUP[(p, algo)]["iterations"] for p in group_probs if LOOKUP[(p, algo)]["converged"]]
            if not vals:
                continue
            means.append(np.mean(vals))
            names.append(label)
            for fam, members in FAMILIES.items():
                if algo in members:
                    colors_list.append(FAMILY_COLORS[fam])
                    break

        if not names:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.7), 5))
        x = np.arange(len(names))
        ax.bar(x, means, color=colors_list, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=50, ha="right", fontsize=8)
        ax.set_ylabel("Avg Iterations")
        ax.set_title(f"Avg Iterations — {group_name.replace('_', ' ').title()} Problems", fontsize=12)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(DIR_PERPROBLEM / f"bar_{group_name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("  [done] small vs large dim bars")


# ── 12. Line search comparison within Newton and BFGS ────────────────────────

def plot_linesearch_comparison():
    """Side-by-side comparison of Backtracking vs Wolfe for Newton/BFGS/DFP."""
    pairs = [
        ("Newton", "Newton_Backtracking", "Newton_Wolfe"),
        ("BFGS", "BFGS_Backtracking", "BFGS_Wolfe"),
        ("DFP", "DFP_Backtracking", "DFP_Wolfe"),
    ]

    for metric, ylabel in [("iterations", "Iterations"), ("cpu_time_s", "CPU Time (s)")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        for ax, (name, bt, wf) in zip(axes, pairs):
            probs_shown = []
            bt_vals, wf_vals = [], []
            for p in PROBLEMS:
                r_bt = LOOKUP.get((p, bt))
                r_wf = LOOKUP.get((p, wf))
                if r_bt and r_bt["converged"] and r_wf and r_wf["converged"]:
                    probs_shown.append(SHORT_PROB.get(p, p))
                    bt_vals.append(r_bt[metric])
                    wf_vals.append(r_wf[metric])

            if not probs_shown:
                ax.set_visible(False)
                continue

            x = np.arange(len(probs_shown))
            w = 0.35
            ax.bar(x - w / 2, bt_vals, w, label="Backtracking", color="#3b82f6", alpha=0.8)
            ax.bar(x + w / 2, wf_vals, w, label="Wolfe", color="#f97316", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(probs_shown, rotation=55, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(name, fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Backtracking vs Wolfe — {ylabel}", fontsize=13, y=1.02)
        fig.tight_layout()
        safe = metric.replace("_", "")
        fig.savefig(DIR_LS / f"linesearch_cmp_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("  [done] line search comparison")


# ── 13. BFGS variant comparison (the core research question) ────────────────

def plot_bfgs_variants():
    """Detailed comparison of all BFGS variants across all problems."""
    bfgs_algos = ["BFGS_Wolfe", "D-BFGS_Wolfe", "DD-BFGS_Wolfe", "C-BFGS_Wolfe", "L-BFGS_Wolfe", "D-L-BFGS_Wolfe"]
    bfgs_labels = {
        "BFGS_Wolfe": "BFGS", "D-BFGS_Wolfe": "D-BFGS",
        "DD-BFGS_Wolfe": "DD-BFGS", "C-BFGS_Wolfe": "C-BFGS",
        "L-BFGS_Wolfe": "L-BFGS", "D-L-BFGS_Wolfe": "D-L-BFGS",
    }
    bfgs_colors = {
        "BFGS_Wolfe": "#ef4444", "D-BFGS_Wolfe": "#f97316",
        "DD-BFGS_Wolfe": "#fbbf24", "C-BFGS_Wolfe": "#22c55e",
        "L-BFGS_Wolfe": "#3b82f6", "D-L-BFGS_Wolfe": "#8b5cf6",
    }

    for metric in ["iterations", "nfev", "cpu_time_s"]:
        _plot_performance_profile(
            metric, bfgs_algos,
            labels=bfgs_labels,
            colors=bfgs_colors,
            title_suffix=" — BFGS Variants",
            fname_suffix="_bfgs_variants",
            tau_max=20.0,
        )

    # per-problem grouped bars
    for problem in PROBLEMS:
        algos_here, vals_here, colors_here, labels_here = [], [], [], []
        for a in bfgs_algos:
            r = LOOKUP.get((problem, a))
            if r and r["converged"]:
                algos_here.append(a)
                vals_here.append(r["iterations"])
                colors_here.append(bfgs_colors[a])
                labels_here.append(bfgs_labels[a])

        if len(algos_here) < 2:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(labels_here))
        ax.bar(x, vals_here, color=colors_here, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_here, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Iterations")
        ax.set_title(f"BFGS Variants — {SHORT_PROB.get(problem, problem)}", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        safe = problem.replace("-", "_").replace(" ", "_")
        fig.savefig(DIR_BFGS / f"bfgs_bar_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("  [done] BFGS variant comparison")


# ── 14. Quasi-Newton vs Second-Order comparison ─────────────────────────────

def plot_qn_vs_second_order():
    """Performance profile: quasi-Newton methods vs true second-order methods."""
    second_order = ["Newton_Wolfe", "TR-Newton-CG"]
    quasi_newton = ["BFGS_Wolfe", "L-BFGS_Wolfe", "D-BFGS_Wolfe", "D-L-BFGS_Wolfe", "DFP_Backtracking"]
    combined = second_order + quasi_newton

    combined_labels = {
        "Newton_Wolfe": "Newton", "TR-Newton-CG": "TR-Newton-CG",
        "BFGS_Wolfe": "BFGS", "L-BFGS_Wolfe": "L-BFGS",
        "D-BFGS_Wolfe": "D-BFGS", "D-L-BFGS_Wolfe": "D-L-BFGS",
        "DFP_Backtracking": "DFP",
    }
    combined_colors = {
        "Newton_Wolfe": "#22c55e", "TR-Newton-CG": "#f97316",
        "BFGS_Wolfe": "#ef4444", "L-BFGS_Wolfe": "#3b82f6",
        "D-BFGS_Wolfe": "#fbbf24", "D-L-BFGS_Wolfe": "#8b5cf6",
        "DFP_Backtracking": "#ec4899",
    }

    for metric in ["iterations", "cpu_time_s"]:
        _plot_performance_profile(
            metric, combined,
            labels=combined_labels,
            colors=combined_colors,
            title_suffix=" — QN vs 2nd Order",
            fname_suffix="_qn_vs_2nd",
            tau_max=30.0,
        )

    print("  [done] QN vs second-order profiles")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Generating plots...")
    plot_global_performance_profiles()
    plot_family_performance_profiles()
    plot_representative_profiles()
    plot_convergence_heatmap()
    plot_iteration_heatmap()
    plot_per_problem_bars()
    plot_iter_vs_cpu()
    plot_eval_breakdown()
    plot_family_summary_bars()
    plot_convergence_rate()
    plot_small_vs_large()
    plot_linesearch_comparison()
    plot_bfgs_variants()
    plot_qn_vs_second_order()
    print(f"\nAll plots saved to {OUTDIR.resolve()}/")


if __name__ == "__main__":
    main()
