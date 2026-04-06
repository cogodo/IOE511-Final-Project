import numpy as np
import scipy.io as sio
from objectives.base import SolverObjective
from algorithms.base import SolverAlgorithm
from options.base import SolverOptions, LineSearchOptions, CGOptions, TrustRegionOptions, LBFGSOptions
from optSolver import optSolver

quad2_data = sio.loadmat('objectives/data/quadratic2.mat')
quad10_data = sio.loadmat('objectives/data/quadratic10.mat')

# constants
alpha_bar = 1.0
c1 = 1e-3
tau = 0.5
max_iters = 1000
epsilon = 1e-5
epsilon_sy = 1e-5

# set up the quad2 problem
quad2_problem = SolverObjective(name='Quadratic', x0=quad2_data['x_0'], A=quad2_data['A'], b=quad2_data['b'], c=quad2_data['c'])

# set up the rosenbrock problem
rosen_problem = SolverObjective(name='Rosenbrock', x0=np.array([[1.2], [1.2]]))

# set up constant gradient descent method and options
GD_const_method = SolverAlgorithm(name="GradientDescent")
GD_const_options = SolverOptions(line_search=LineSearchOptions(method='Constant', const_alpha=1e-3), max_iterations=max_iters, term_tol=epsilon)

# set up backtracking gradient descent method and options
GD_backtracking_method = SolverAlgorithm(name='GradientDescent')
GD_backtracking_options = SolverOptions(line_search=LineSearchOptions(method='Backtracking', alpha0=alpha_bar, c1=c1, tau=tau), max_iterations=max_iters, term_tol=epsilon)

# set up Wolfe gradient descent method and options
GD_wolfe_method = SolverAlgorithm(name='GradientDescent')
GD_wolfe_options = SolverOptions(line_search=LineSearchOptions(method='Wolfe'), max_iterations=max_iters, term_tol=epsilon)

# set up the backtracking newton method and options
newton_backtracking_method = SolverAlgorithm(name='Newton')
newton_backtracking_options = SolverOptions(line_search=LineSearchOptions(method='Backtracking', alpha0=alpha_bar, c1=c1, tau=tau), max_iterations=max_iters, term_tol=epsilon)

# set up Wolfe newton method and options
newton_wolfe_method = SolverAlgorithm(name='Newton')
newton_wolfe_options = SolverOptions(line_search=LineSearchOptions(method='Wolfe'), max_iterations=max_iters, term_tol=epsilon)

# set up Wolfe BFGS method and options
bfgs_wolfe_method = SolverAlgorithm(name='BFGS')
bfgs_wolfe_options = SolverOptions(line_search=LineSearchOptions(method='Wolfe', c1=c1), max_iterations=max_iters, term_tol=epsilon, sy_tol=epsilon_sy)

# set up Wolfe LBFGS method and options
lbfgs_wolfe_method = SolverAlgorithm(name='L-BFGS')
lbfgs_wolfe_options = SolverOptions(line_search=LineSearchOptions(method='Wolfe', c1=c1), max_iterations=max_iters,
                                    term_tol=epsilon, sy_tol=epsilon_sy, lbfgs=LBFGSOptions(history_length=2))
# run quad2 problem with GD
# x, f = optSolver(problem=quad2_problem, method=GD_const_method, options=GD_const_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=quad2_problem, method=GD_backtracking_method, options=GD_backtracking_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=quad2_problem, method=GD_wolfe_method, options=GD_wolfe_options)
# print(f'x: {x}, f: {f}')
# # # run rosenbrock with GD
# x, f = optSolver(problem=rosen_problem, method=GD_backtracking_method, options=GD_backtracking_options)
# print(f'x: {x}, f: {f}')

# run quad2 problem with Newton
x, f = optSolver(problem=quad2_problem, method=newton_backtracking_method, options=newton_backtracking_options)
print(f'x: {x}, f: {f}')

x, f = optSolver(problem=quad2_problem, method=newton_wolfe_method, options=newton_wolfe_options)
print(f'x: {x}, f: {f}')

# run rosenbrock with Newton
x, f = optSolver(problem=rosen_problem, method=newton_backtracking_method, options=newton_backtracking_options)
print(f'x: {x}, f: {f}')
x, f = optSolver(problem=rosen_problem, method=newton_wolfe_method, options=newton_wolfe_options)
print(f'x: {x}, f: {f}')

# run rosenbrock with BFGS
# x, f = optSolver(problem=rosen_problem, method=bfgs_wolfe_method, options=bfgs_wolfe_options)
# print(f'x: {x}, f: {f}')

#run rosenbrock with LBFGS
# x, f = optSolver(problem=rosen_problem, method=lbfgs_wolfe_method, options=lbfgs_wolfe_options)

# print(f'x: {x}, f: {f}')
# TODO: more things to track (with plots hopefully) - num iterations to converge, time to converge, total memory(?)
# put multiple algos on the same plot when it makes sense to compare for the paper / poster
# for individual runs, we probably want 2 loss plots (I think): One in terms of iterations, and one in terms of time
# or if the time one doesnt turn out we could just output the total time taken at the end too.
