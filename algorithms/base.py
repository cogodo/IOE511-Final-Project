# Class to make the algorithm-level abstraction clear

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

from algorithms.utils import StepResults, InternalAlgorithmState


# NOTE: needed to set frozen to false here to run without error
@dataclass(frozen=False, slots=True)
class SolverAlgorithm:
    name: str
    step: Callable[..., StepResults] = None

