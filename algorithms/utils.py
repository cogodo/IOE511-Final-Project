from dataclasses import dataclass
from objectives.base import SolverObjective
from options.base import SolverOptions
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class StepResults:
    x_new: Array = None
    f_new: Array = None
    g_new: Array = None
    H_new: Array = None
    Hinv_approx_new: Array = None
    d: Array = None
    alpha: float = None

def backtracking_line_search(x: Array, f: Array, g: Array, d: Array, objective: SolverObjective, options: SolverOptions):
    alpha = options.line_search.alpha0

    # perform backtracking line search
    while objective.value(x + alpha*d) > f + options.line_search.c1*alpha*g.transpose() @ d:
        alpha = alpha*options.line_search.tau

    return alpha

def weak_wolfe_line_search(x: Array, f: Array, g: Array, d: Array, objective: SolverObjective, options: SolverOptions):
    alpha = options.line_search.alpha0
    alpha_low = options.line_search.alpha_low0
    alpha_high = options.line_search.alpha_high0

    # perform weak Wolfe line search
    while True:
        if (objective.value(x + alpha*d) <= f + options.line_search.c1*alpha*g.transpose() @ d):
            if (objective.grad(x + alpha*d).transpose() @ d >= options.line_search.c2*g.transpose() @ d):
                break
            alpha_low = alpha
        else:
            alpha_high = alpha
        
        alpha = options.line_search.c*alpha_low + (1 - options.line_search.c)*alpha_high

    return alpha