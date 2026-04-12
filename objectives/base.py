# Class to define the problem clearly

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

# NOTE: needed to set frozen to False here for program to run without error
@dataclass(frozen=False, slots=True)
class SolverObjective:
    name: str
    x0: Array
    A: Array = None
    b: Array = None
    c: Array = None
    value: Callable[[Array], float] = None
    grad: Callable[[Array], Array] = None
    hess: Callable[[Array], Array] = None
    f_star: float = None
