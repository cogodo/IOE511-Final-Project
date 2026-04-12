from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LineSearchOptions:
    method: str = "Backtracking"
    c1: float = 1e-4
    c2: float = 0.9
    alpha0: float = 1.0
    const_alpha: float = 1e-3
    alpha_low0: float = 0.0
    alpha_high0: float = 1000.0
    rho: float = 0.5
    tau: float = 0.5
    c: float = 0.5


@dataclass(frozen=True, slots=True)
class TrustRegionOptions:
    c1: float = 0.25
    c2: float = 0.75
    delta_init: float = 1.0


@dataclass(frozen=True, slots=True)
class CGOptions:
    term_tol: float = 1e-5
    max_iterations: int = 100


@dataclass(frozen=True, slots=True)
class NewtonOptions:
    cholesky_beta: float = 1e-5


@dataclass(frozen=True, slots=True)
class BFGSVariantOptions:
    sy_tol: float = 1e-6
    history_length: int = 10
    Hinv_approx_init: Array = None
    H_approx_init: Array = None
    cautious_tol: float = 1e-5
    cautious_alpha: int = 2
    sr1_tol: float = 1e-5


@dataclass(frozen=True, slots=True)
class SolverOptions:
    term_tol: float = 1e-6
    max_iterations: int = 1000
    line_search: LineSearchOptions = LineSearchOptions()
    trust_region: TrustRegionOptions = TrustRegionOptions()
    cg: CGOptions = CGOptions()
    newton: NewtonOptions = NewtonOptions()
    bfgs: BFGSVariantOptions = BFGSVariantOptions()