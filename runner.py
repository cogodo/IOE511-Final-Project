import numpy as np
import scipy.io as sio
from objectives.base import SolverObjective
from algorithms.base import SolverAlgorithm
from options.base import SolverOptions, LineSearchOptions, CGOptions, TrustRegionOptions, BFGSVariantOptions
from optSolver import optSolver
import matplotlib.pyplot as plt

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
rosen_2_problem = SolverObjective(name='Rosenbrock-2', x0=np.array([[-1.2], [1]]), f_star=0.0)

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

# set up backtracking BFGS method and options
bfgs_back_method = SolverAlgorithm(name='BFGS')
bfgs_back_options = SolverOptions(line_search=LineSearchOptions(method='Backtracking', alpha0=alpha_bar, c1=c1, tau=tau), bfgs=BFGSVariantOptions(sy_tol=epsilon_sy), max_iterations=max_iters, term_tol=epsilon)

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

# set up backtracking DFP method and options
dfp_back_method = SolverAlgorithm(name='DFP')
dfp_back_options = SolverOptions(line_search=LineSearchOptions(method='Backtracking', alpha0=alpha_bar, c1=c1, tau=tau), bfgs=BFGSVariantOptions(sy_tol=epsilon_sy), max_iterations=max_iters, term_tol=epsilon)

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

method_options = {
    'GradientDescent_Constant': GD_const_options,
    'GradientDescent_Backtracking': GD_backtracking_options,
    'GradientDescent_Wolfe': GD_wolfe_options,
    'Newton_Backtracking': newton_backtracking_options,
    'Newton_Wolfe': newton_wolfe_options,
    'BFGS_Backtracking': bfgs_back_options,
    'BFGS_Wolfe': bfgs_wolfe_options,
    'TR-Newton-CG': tr_newton_cg_options,
    'TR-SR1-CG': tr_sr1_cg_options,
    'DFP_Backtracking': dfp_back_options,
    'DFP_Wolfe': dfp_wolfe_options
}

def plot_iterations_v_f_residual_all_canon_methods(problem: SolverObjective, title: str, method_options: dict[str, SolverOptions]):
    
    gd_const_f_vals = []
    gd_back_f_vals = []
    gd_wolfe_f_vals = []
    newton_back_f_vals = []
    newton_wolfe_f_vals = []
    tr_newton_cg_f_vals = []
    tr_sr1_cg_f_vals = []
    bfgs_back_f_vals = []
    bfgs_wolfe_f_vals = []
    dfp_back_f_vals = []
    dfp_wolfe_f_vals = []

    # gradient descent
    _, _ = optSolver(problem=problem, method=SolverAlgorithm('GradientDescent'), options=method_options['GradientDescent_Constant'], f_vals=gd_const_f_vals)
    _, _ = optSolver(problem=problem, method=SolverAlgorithm('GradientDescent'), options=method_options['GradientDescent_Backtracking'], f_vals=gd_back_f_vals)
    _, _ = optSolver(problem=problem, method=SolverAlgorithm('GradientDescent'), options=method_options['GradientDescent_Wolfe'], f_vals=gd_wolfe_f_vals)

    # newton
    _, _ = optSolver(problem=problem, method=SolverAlgorithm('Newton'), options=method_options['Newton_Backtracking'], f_vals=newton_back_f_vals)
    _, _ = optSolver(problem=problem, method=SolverAlgorithm('Newton'), options=method_options['Newton_Backtracking'], f_vals=newton_wolfe_f_vals)

    # trust region
    _, _ = optSolver(problem=problem, method=SolverAlgorithm('TR-Newton-CG'), options=method_options['TR-Newton-CG'], f_vals=tr_newton_cg_f_vals)
    _, _ = optSolver(problem=problem, method=SolverAlgorithm('TR-SR1-CG'), options=method_options['TR-SR1-CG'], f_vals=tr_sr1_cg_f_vals)

    # # BFGS
    _, _ = optSolver(problem=problem, method=SolverAlgorithm('BFGS'), options=method_options['BFGS_Backtracking'], f_vals=bfgs_back_f_vals)
    _, _ = optSolver(problem=problem, method=SolverAlgorithm('BFGS'), options=method_options['BFGS_Wolfe'], f_vals=bfgs_wolfe_f_vals)

    # # DFP
    _, _ = optSolver(problem=problem, method=SolverAlgorithm('DFP'), options=method_options['DFP_Backtracking'], f_vals=dfp_back_f_vals)
    _, _ = optSolver(problem=problem, method=SolverAlgorithm('DFP'), options=method_options['DFP_Wolfe'], f_vals=dfp_wolfe_f_vals)
    
    blues = plt.get_cmap('Blues')
    blue_shades = [blues(0.4), blues(0.7), blues(1.0)] 

    greens = plt.get_cmap('Greens')
    green_shades = [greens(0.4), greens(0.8)]

    oranges = plt.get_cmap('Oranges')
    orange_shades = [oranges(0.4), oranges(0.7)]

    reds = plt.get_cmap('RdPu')
    red_shades = [reds(0.4), reds(0.8)]

    purples = plt.get_cmap('Purples')
    purple_shades = [purples(0.4), purples(0.8)]

    plt.plot(np.array(gd_const_f_vals) - problem.f_star, color=blue_shades[0], label='GD (constant step size)')
    plt.plot(np.array(gd_back_f_vals) - problem.f_star, color=blue_shades[1], label='GD (backtracking)')
    plt.plot(np.array(gd_wolfe_f_vals) - problem.f_star, color=blue_shades[2], label='GD (Wolfe)')
    plt.plot(np.array(newton_back_f_vals) - problem.f_star, color=green_shades[0], label='Newton (backtracking)')
    plt.plot(np.array(newton_wolfe_f_vals) - problem.f_star, color=green_shades[1], label='Newton (Wolfe)')
    plt.plot(np.array(tr_newton_cg_f_vals) - problem.f_star, color=orange_shades[0], label='TR (Newton)')
    plt.plot(np.array(tr_sr1_cg_f_vals) - problem.f_star, color=orange_shades[1], label='TR (SR1)')
    plt.plot(np.array(bfgs_back_f_vals) - problem.f_star, color=red_shades[0], label='BFGS (backtracking)')
    plt.plot(np.array(bfgs_wolfe_f_vals) - problem.f_star, color=red_shades[1], label='BFGS (Wolfe)')
    plt.plot(np.array(dfp_back_f_vals) - problem.f_star, color=purple_shades[0], label='DFP (backtracking)')
    plt.plot(np.array(dfp_wolfe_f_vals) - problem.f_star, color=purple_shades[1], label='DFP (Wolfe)')

    plt.yscale('log')
    plt.xlabel('iterations k')
    plt.ylabel('f - f*')

    plt.title(title)
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4, fontsize=6)
    plt.tight_layout()
    plt.savefig(f'plots/{title}.png')

plot_iterations_v_f_residual_all_canon_methods(problem=rosen_2_problem, title='test', method_options=method_options)
    
    










