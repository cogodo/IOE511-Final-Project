# Class to make the algorithm-level abstraction clear

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from algorithms.utils import LineSearchOptions, TrustRegionOptions, CGOptions

Array = npt.NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class SolverAlgorithm:
    name: str
    line_search: LineSearchOptions | None = None
    trust_region: TrustRegionOptions | None = None
    cg: CGOptions | None = None


