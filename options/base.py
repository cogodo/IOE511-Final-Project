from dataclasses import dataclass

# NOTE: what is rho used for?
# alpha_bar: constant step size
@dataclass(frozen=True, slots=True)
class LineSearchOptions:
    method: str = "Backtracking"
    c1: float = 1e-4
    c2: float = 0.9
    alpha0: float = 1.0
    rho: float = 0.5
    const_alpha: float = 1e-3
    tau: float = 1.0
    
@dataclass(frozen=True, slots=True)
class TrustRegionOptions:
    eta1: float = 0.25
    eta2: float = 0.75

@dataclass(frozen=True, slots=True)
class CGOptions:
    term_tol: float = 1e-6
    max_iterations: int = 100

@dataclass(frozen=True, slots=True)
class SolverOptions:
    term_tol: float = 1e-6
    max_iterations: int = 1000
    line_search: LineSearchOptions = LineSearchOptions()
    trust_region: TrustRegionOptions = TrustRegionOptions()
    cg: CGOptions = CGOptions
    