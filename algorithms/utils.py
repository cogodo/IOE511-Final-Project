from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class StepResults:
    x_new: Array = None
    f_new: Array = None
    g_new: Array = None
    H_new: Array = None
    d: Array = None
    alpha: float = None
