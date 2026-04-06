from dataclasses import dataclass
from objectives.base import SolverObjective
from options.base import SolverOptions
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class InternalAlgorithmState:
    pass

@dataclass(frozen=True, slots=True)
class StepResults:
    x_new: Array = None
    f_new: Array = None
    g_new: Array = None
    H_new: Array = None
    Hinv_approx_new: Array = None
    d: Array = None
    alpha: float = None
    internal_state: InternalAlgorithmState = None

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


class VectorCircularBuffer:
    def __init__(self, capacity, vector_size, dtype=np.float64):
        self.capacity = capacity
        self.vector_size = vector_size
        # Shape is (Capacity, Vector Size)
        self.buffer = np.zeros((capacity, vector_size), dtype=dtype)
        self.index = 0
        self.full = False

    def append(self, data: Array):
        """
        Adds one or more vectors.
        'data' should have shape (vector_size,) or (N, vector_size).
        """
        data = np.atleast_2d(data)
        n = data.shape[0]

        if n > self.capacity:
            data = data[-self.capacity:]
            n = self.capacity

        end = (self.index + n) % self.capacity

        if self.index + n <= self.capacity:
            # Standard block write
            self.buffer[self.index:self.index + n, :] = data
        else:
            # Wrap-around write
            pivot = self.capacity - self.index
            self.buffer[self.index:, :] = data[:pivot, :]
            self.buffer[:end, :] = data[pivot:, :]

        self.index = end
        if n >= self.capacity or self.index == 0:
            self.full = True

    def get_ordered(self):
        """Returns the vectors in chronological order."""
        if not self.full:
            return self.buffer[:self.index].copy()
        # Stack the 'old' tail and 'new' head along the first axis
        return np.vstack((self.buffer[self.index:], self.buffer[:self.index]))

    @property
    def latest(self):
        """Returns the most recently added vector."""
        return self.buffer[(self.index - 1) % self.capacity]

    def iter_chunks(self):
        """
        Yields 1 or 2 views (not copies) of the buffer in chronological order.
        Usage:
            for chunk in buf.iter_chunks():
                process(chunk)
        """
        if not self.full:
            yield self.buffer[:self.index]
        else:
            # Yield from current index to the end (oldest data)
            yield self.buffer[self.index:]
            # Yield from start to current index (newest data)
            if self.index > 0:
                yield self.buffer[:self.index]

    def __iter__(self):
        """
        Yields individual vectors one by one without copying the whole buffer.
        """
        for chunk in self.iter_chunks():
            for vector in chunk:
                yield vector

@dataclass(frozen=True, slots=True)
class LBFGSState(InternalAlgorithmState):
    s_buffer: VectorCircularBuffer
    y_buffer: VectorCircularBuffer

def two_loop_recursion(g: Array, Hinv_approx_init: Array, s_buffer: VectorCircularBuffer, y_buffer: VectorCircularBuffer):
    q = np.squeeze(np.copy(g))

    s_array = s_buffer.get_ordered()
    y_array = y_buffer.get_ordered()

    inner_products = np.einsum('ij,ij->i', s_array, y_array)
    rho = 1 / inner_products
    alphas = np.zeros_like(inner_products)
    betas = np.zeros_like(inner_products)

    m = s_array.shape[0]
    loop_indices = np.flip(np.arange(m))

    for ii in loop_indices:
        si = s_array[ii]
        yi = y_array[ii]
        alphas[ii] = rho[ii] * np.dot(si, q)
        q = q-alphas[ii]*yi

    gamma_k = 1
    r = gamma_k * Hinv_approx_init @ q

    for ii in np.flip(loop_indices):
        si = s_array[ii]
        yi = y_array[ii]
        betas[ii] = rho[ii]* np.dot(yi, r)
        r = r + si*(alphas[ii] - betas[ii])

    return -r[:, None]