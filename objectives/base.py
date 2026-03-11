# Class to define the problem clearly

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class Objective:
    name: str
    x0: Array
    value: Callable[[Array], float]
    grad: Callable[[Array], Array]
    hess: Callable[[Array], Array]
