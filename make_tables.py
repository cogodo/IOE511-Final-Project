"""Generate LaTeX summary tables from benchmark_results.json."""
import json
from pathlib import Path

data = json.loads(Path("results/benchmark_results.json").read_text())

problems = list(dict.fromkeys(r["problem"] for r in data))
algorithms = list(dict.fromkeys(r["algorithm"] for r in data))

lookup = {(r["problem"], r["algorithm"]): r for r in data}

SHORT = {
    "GD_Constant": "GD-C",
    "GD_Backtracking": "GD-B",
    "GD_Wolfe": "GD-W",
    "Newton_Backtracking": "N-B",
    "Newton_Wolfe": "N-W",
    "TR-Newton-CG": "TR-N",
    "TR-SR1-CG": "TR-S",
    "BFGS_Backtracking": "B-B",
    "BFGS_Wolfe": "B-W",
    "D-BFGS_Wolfe": "D-B",
    "DD-BFGS_Wolfe": "DD-B",
    "C-BFGS_Wolfe": "C-B",
    "L-BFGS_Wolfe": "L-B",
    "D-L-BFGS_Wolfe": "DL-B",
    "DFP_Backtracking": "DF-B",
    "DFP_Wolfe": "DF-W",
}

SHORT_PROB = {
    "quad_10_10": "Q-10/10",
    "quad_10_1000": "Q-10/1k",
    "quad_1000_10": "Q-1k/10",
    "quad_1000_1000": "Q-1k/1k",
    "quartic_1": "Quartic-1",
    "quartic_2": "Quartic-2",
    "Rosenbrock-2": "Rosen-2",
    "Rosenbrock-100": "Rosen-100",
    "datafit_2": "DataFit",
    "exp_10": "Exp-10",
    "exp_1000": "Exp-1000",
    "genhumps_5": "GenHumps",
}


def make_table(metric: str, caption: str, label: str, fmt: str = "d"):
    n = len(algorithms)
    col_spec = "l" + "r" * n
    header_cells = " & ".join(f"\\romark{{{SHORT[a]}}}" for a in algorithms)

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

    for p in problems:
        cells = [SHORT_PROB[p]]
        for a in algorithms:
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

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "}",
        r"\end{table}",
    ]
    return "\n".join(lines)


preamble = r"""\newcommand{\romark}[1]{\rotatebox{70}{\makebox[0pt][l]{#1}}}"""

tables = [
    make_table("iterations", "Summary of Results: Number of Iterations.", "tab:iters"),
    make_table("nfev", "Summary of Results: Number of Function Evaluations.", "tab:nfev"),
    make_table("ngev", "Summary of Results: Number of Gradient Evaluations.", "tab:ngev"),
    make_table("cpu_time_s", "Summary of Results: CPU Seconds.", "tab:cpu", fmt="f"),
]

output = preamble + "\n\n" + "\n\n".join(tables)
Path("results/summary_tables.tex").write_text(output)
print(f"Written to results/summary_tables.tex ({len(output)} chars)")
