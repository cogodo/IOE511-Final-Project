import numpy as np
import scipy.io as sio
from objectives.base import SolverObjective
from algorithms.base import SolverAlgorithm
from algorithms.utils import LineSearchOptions, TrustRegionOptions, CGOptions, SolverOptions
from optSolver import optSolver

quad2_data = sio.loadmat('objectives/data/quadratic2.mat')
quad10_data = sio.loadmat('objectives/data/quadratic10.mat')

# constants
alpha_bar = 1.0
c1 = 1e-3
tau = 0.5
max_iters = 100
epsilon = 1e-5

# set up the quad2 problem
quad2_problem = SolverObjective(name='Quadratic', x0=quad2_data['x_0'], A=quad2_data['A'], b=quad2_data['b'], c=quad2_data['c'])

# set up the rosenbrock problem
rosen_problem = SolverObjective(name='Rosenbrock', x0=np.array([[1.2], [1.2]]))

# set up constant gradient descent method
GD_const_method = SolverAlgorithm(name="GradientDescent", line_search=LineSearchOptions(method='Constant', const_alpha=1e-3))

# set up backtracking gradient descent method
GD_backtracking_method = SolverAlgorithm(name='GradientDescent', line_search=LineSearchOptions(method='Backtracking', alpha0=alpha_bar, c1=c1, tau=tau))

# set up the backtracking newton method
newton_backtracking_method = SolverAlgorithm(name='Newton', line_search=LineSearchOptions(method='Backtracking', alpha0=alpha_bar, c1=c1, tau=tau))

# set options
options = SolverOptions(term_tol=epsilon, max_iterations=max_iters)

# run quad2 problem with GD
# x, f = optSolver(problem=quad2_problem, method=GD_const_method, options=options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=quad2_problem, method=GD_backtracking_method, options=options)
# print(f'x: {x}, f: {f}')

# run rosenbrock with GD
x, f = optSolver(problem=rosen_problem, method=GD_backtracking_method, options=options)
print(f'x: {x}, f: {f}')

# run quad2 problem with Newton
x, f = optSolver(problem=quad2_problem, method=newton_backtracking_method, options=options)
print(f'x: {x}, f: {f}')

# run rosenbrock with Newton
x, f = optSolver(problem=rosen_problem, method=newton_backtracking_method, options=options)
print(f'x: {x}, f: {f}')