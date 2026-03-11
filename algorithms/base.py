# Class to make the algorithm-level abstraction clear

from dataclasses import dataclass
from typing import Callable
import numpy as np
import numpy.typing as npt

from algorithms.utils import Options
from objectives.base import Objective

Array = npt.NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class Algorithm:
    name: str
    step: Callable[Array, Array]


