import numpy as np
import scipy.io as sio
from objectives.base import SolverObjective
from algorithms.base import SolverAlgorithm
from options.base import SolverOptions, LineSearchOptions, CGOptions, TrustRegionOptions, BFGSVariantOptions
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

# set up the rosenbrock-2 problem
rosen_2_problem = SolverObjective(name='Rosenbrock-2', x0=np.array([[-1.2], [1]]))

# set up the rosenbrock-100 problem
rosen_100_starting_point = np.ones(shape=(100,1))
rosen_100_starting_point[0] = -1.2
rosen_100_problem = SolverObjective(name='Rosenbrock-100', x0=rosen_100_starting_point)

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

# set up TR Newton method with CG subproblem solver
tr_newton_cg_method = SolverAlgorithm(name='TR-Newton-CG')
tr_newton_cg_options = SolverOptions()

# set up TR SR1 method with CG subproblem solver
tr_sr1_cg_method = SolverAlgorithm(name='TR-SR1-CG')
tr_sr1_cg_options = SolverOptions()

# set up Wolfe BFGS method and options
bfgs_wolfe_method = SolverAlgorithm(name='BFGS')
bfgs_wolfe_options = SolverOptions(line_search=LineSearchOptions(method='Wolfe', c1=c1), bfgs=BFGSVariantOptions(sy_tol=epsilon_sy), max_iterations=max_iters, term_tol=epsilon)

# set up Wolfe DBFGS method and options
dbfgs_wolfe_method = SolverAlgorithm(name='D-BFGS')
dbfgs_wolfe_options = SolverOptions(line_search=LineSearchOptions(method='Wolfe', c1=c1), bfgs=BFGSVariantOptions(sy_tol=epsilon_sy), max_iterations=max_iters, term_tol=epsilon)

# set up Wolfe DDBFGS method and options
ddbfgs_wolfe_method = SolverAlgorithm(name='DD-BFGS')
ddbfgs_wolfe_options = SolverOptions(line_search=LineSearchOptions(method='Wolfe', c1=c1), bfgs=BFGSVariantOptions(sy_tol=epsilon_sy), max_iterations=max_iters, term_tol=epsilon)

# set up Wolfe CBFGS method and options
cbfgs_wolfe_method = SolverAlgorithm(name='C-BFGS')
cbfgs_wolfe_options = SolverOptions(line_search=LineSearchOptions(method='Wolfe', c1=c1), bfgs=BFGSVariantOptions(sy_tol=epsilon_sy), max_iterations=max_iters, term_tol=epsilon)

# set up Wolfe LBFGS method and options
lbfgs_wolfe_method = SolverAlgorithm(name='L-BFGS')
lbfgs_wolfe_options = SolverOptions(line_search=LineSearchOptions(method='Wolfe', c1=c1), max_iterations=max_iters,
                                    term_tol=epsilon, bfgs=BFGSVariantOptions(history_length=2, sy_tol=epsilon_sy))

# set up Wolfe DFP method and options
dfp_wolfe_method = SolverAlgorithm(name='DFP')
dfp_wolfe_options = SolverOptions(line_search=LineSearchOptions(method='Wolfe', c1=c1), bfgs=BFGSVariantOptions(sy_tol=epsilon_sy), max_iterations=max_iters, term_tol=epsilon)

# set up Wolfe Damped LBFGS method and options
dlbfgs_wolfe_method = SolverAlgorithm(name='D-L-BFGS')
dlbfgs_wolfe_options = SolverOptions(line_search=LineSearchOptions(method='Wolfe', c1=c1), max_iterations=max_iters,
                                    term_tol=epsilon, bfgs=BFGSVariantOptions(history_length=2, sy_tol=epsilon_sy))
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
# x, f = optSolver(problem=quad2_problem, method=newton_backtracking_method, options=newton_backtracking_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=quad2_problem, method=newton_wolfe_method, options=newton_wolfe_options)
# print(f'x: {x}, f: {f}')

# run rosenbrock-2 with all methods
# x, f = optSolver(problem=rosen_2_problem, method=GD_backtracking_method, options=GD_backtracking_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_2_problem, method=GD_wolfe_method, options=GD_wolfe_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_2_problem, method=newton_backtracking_method, options=newton_backtracking_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_2_problem, method=newton_wolfe_method, options=newton_wolfe_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_2_problem, method=tr_newton_cg_method, options=tr_newton_cg_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_2_problem, method=tr_sr1_cg_method, options=tr_newton_cg_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_2_problem, method=bfgs_wolfe_method, options=bfgs_wolfe_options)
# print(f'x: {x}, f: {f}')
x, f = optSolver(problem=rosen_2_problem, method=dbfgs_wolfe_method, options=dbfgs_wolfe_options)
print(f'x: {x}, f: {f}')
x, f = optSolver(problem=rosen_2_problem, method=ddbfgs_wolfe_method, options=ddbfgs_wolfe_options)
print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_2_problem, method=cbfgs_wolfe_method, options=cbfgs_wolfe_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_2_problem, method=lbfgs_wolfe_method, options=lbfgs_wolfe_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_2_problem, method=dfp_wolfe_method, options=dfp_wolfe_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_2_problem, method=dlbfgs_wolfe_method, options=dlbfgs_wolfe_options)
# print(f'x: {x}, f: {f}')

# run rosenbrock-100 with all methods
# x, f = optSolver(problem=rosen_100_problem, method=newton_backtracking_method, options=newton_backtracking_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_100_problem, method=newton_wolfe_method, options=newton_wolfe_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_100_problem, method=bfgs_wolfe_method, options=bfgs_wolfe_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_100_problem, method=dbfgs_wolfe_method, options=dbfgs_wolfe_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_100_problem, method=cbfgs_wolfe_method, options=cbfgs_wolfe_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_100_problem, method=lbfgs_wolfe_method, options=lbfgs_wolfe_options)
# print(f'x: {x}, f: {f}')
# x, f = optSolver(problem=rosen_100_problem, method=dfp_wolfe_method, options=dfp_wolfe_options)
# print(f'x: {x}, f: {f}')

# TODO: more things to track (with plots hopefully) - num iterations to converge, time to converge, total memory(?)
# put multiple algos on the same plot when it makes sense to compare for the paper / poster
# for individual runs, we probably want 2 loss plots (I think): One in terms of iterations, and one in terms of time
# or if the time one doesnt turn out we could just output the total time taken at the end too.
