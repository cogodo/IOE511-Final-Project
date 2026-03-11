import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

from algorithms.utils import SolverOptions
from objectives.base import Objective

def gradient_descent(objective: Objective, x: Array, options: SolverOptions):
    pass

def newton(objective: Objective, x: Array, options: SolverOptions):
    pass

def trsr1cg(objective: Objective, x: Array, options: SolverOptions):
    pass

def sr1(objective: Objective, x: Array, options: SolverOptions):
    pass

def bfgs(objective: Objective, x: Array, options: SolverOptions):
    pass

def dfp(objective: Objective, x: Array, options: SolverOptions):
    pass