from benchmark import run_one, AlgoSpec
from sklearn.model_selection import ParameterGrid

from options.base import LineSearchOptions, SolverOptions, BFGSVariantOptions, TrustRegionOptions
from runner import build_methods, build_problems

import pandas as pd
from tqdm import tqdm

def gridsearch(method_name: str, options: SolverOptions):
    
    problems = build_problems()
    label = method_name

    grid = {}

    if method_name == 'BFGS' or method_name == 'L-BFGS':
        grid['SY_SR_TOL'] = [1e-1, 1e-3, 1e-5, 1e-7]
    
    if method_name == 'L-BFGS' or method_name == 'D-L-BFGS' or method_name == 'C-L-BFGS' or method_name == 'DD-L-BFGS':
        grid['HISTORY_LENGTH'] = [2, 5, 10, 20]

    if method_name == 'C-BFGS' or method_name == 'C-L-BFGS':
        grid['CAUTIOUS_TOL'] = [1e-1, 1e-3, 1e-5, 1e-7]
        grid['CAUTIOUS_ALPHA'] = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    if method_name == 'TR-Newton-CG':
        grid['TR_C1'] = [1e-3, 1e-2, 1e-1, 0.25]
        grid['TR_C2'] = [0.5, 0.75, 0.9]
        grid['INIT_RADIUS'] = [0.1, 0.5, 1.0, 5.0, 10.0]

    if options.line_search.method == 'Backtracking':
        grid['LINESEARCH_C1'] = [1e-4, 1e-3, 1e-2, 0.3]
        grid['LINESEARCH_TAU'] = [0.1, 0.5]
        label += '_BACKTRACKING'
    
    if options.line_search.method == 'Wolfe':
        grid['LINESEARCH_C1'] = [1e-4, 1e-3, 1e-2, 1e-1]
        grid['LINESEARCH_C2'] = [0.8, 0.85, 0.9, 0.95]
        label += '_WOLFE'

    if options.line_search.method == 'Constant':
        grid['LINESEARCH_CONST_ALPHA'] = [1e-4, 1e-3, 1e-2]
        label += '_CONSTANT'


    all_results = []
    for params in tqdm(ParameterGrid(grid)):

        default_options = SolverOptions()

        
        options = SolverOptions(bfgs=BFGSVariantOptions(sy_tol=params.get('SY_SR_TOL', default_options.bfgs.sy_tol),
                                                        sr1_tol=params.get('SY_SR_TOL', default_options.bfgs.sy_tol),
                                                        cautious_alpha=params.get('CAUTIOUS_ALPHA', default_options.bfgs.cautious_alpha),
                                                        cautious_tol=params.get('CAUTIOUS_TOL', default_options.bfgs.cautious_tol),
                                                        history_length=params.get('HISTORY_LENGTH', default_options.bfgs.history_length)),
                                line_search=LineSearchOptions(c1=params.get('LINESEARCH_C1', default_options.line_search.c1),
                                                              c2=params.get('LINESEARCH_C2', default_options.line_search.c2),
                                                              tau=params.get('LINESEARCH_TAU', default_options.line_search.tau),
                                                              const_alpha=params.get('LINESEARCH_CONST_ALPHA', default_options.line_search.const_alpha)),
                                trust_region=TrustRegionOptions(c1=params.get('TR_C1', default_options.trust_region.c1),
                                                                c2=params.get('TR_C2', default_options.trust_region.c2),
                                                                delta_init=params.get('INIT_RADIUS', default_options.trust_region.delta_init)))
        

        run_result = run_one(problems['Rosenbrock-2'], AlgoSpec(label=label + '_ROSENBROCK2', algo_name=method_name, options=options))

        row = params.copy()
        row['CONVERGED'] = run_result.converged
        row['PROBLEM'] = run_result.problem
        row['ITERATIONS'] = run_result.iterations
        row['TIME'] = run_result.cpu_time_s

        all_results.append(row)

        run_result = run_one(problems['Rosenbrock-100'], AlgoSpec(label=label + '_ROSENBROCK100', algo_name=method_name, options=options))

        row = params.copy()
        row['CONVERGED'] = run_result.converged
        row['PROBLEM'] = run_result.problem
        row['ITERATIONS'] = run_result.iterations
        row['TIME'] = run_result.cpu_time_s

        all_results.append(row)

        run_result = run_one(problems['datafit_2'], AlgoSpec(label=label + '_DATAFIT2', algo_name=method_name, options=options))

        row = params.copy()
        row['CONVERGED'] = run_result.converged
        row['PROBLEM'] = run_result.problem
        row['ITERATIONS'] = run_result.iterations
        row['TIME'] = run_result.cpu_time_s

        all_results.append(row)

        run_result = run_one(problems['exp_10'], AlgoSpec(label=label + '_EXP10', algo_name=method_name, options=options))

        row = params.copy()
        row['CONVERGED'] = run_result.converged
        row['PROBLEM'] = run_result.problem
        row['ITERATIONS'] = run_result.iterations
        row['TIME'] = run_result.cpu_time_s

        all_results.append(row)

        run_result = run_one(problems['exp_1000'], AlgoSpec(label=label + '_EXP1000', algo_name=method_name, options=options))

        row = params.copy()
        row['CONVERGED'] = run_result.converged
        row['PROBLEM'] = run_result.problem
        row['ITERATIONS'] = run_result.iterations
        row['TIME'] = run_result.cpu_time_s

        all_results.append(row)

        run_result = run_one(problems['genhumps_5'], AlgoSpec(label=label + '_GENHUMPS5', algo_name=method_name, options=options))

        row = params.copy()
        row['CONVERGED'] = run_result.converged
        row['PROBLEM'] = run_result.problem
        row['ITERATIONS'] = run_result.iterations
        row['TIME'] = run_result.cpu_time_s

        all_results.append(row)

        run_result = run_one(problems['quad_10_10'], AlgoSpec(label=label + '_QUAD10_10', algo_name=method_name, options=options))

        row = params.copy()
        row['CONVERGED'] = run_result.converged
        row['PROBLEM'] = run_result.problem
        row['ITERATIONS'] = run_result.iterations
        row['TIME'] = run_result.cpu_time_s

        all_results.append(row)

        run_result = run_one(problems['quad_10_1000'], AlgoSpec(label=label + '_QUAD10_1000', algo_name=method_name, options=options))

        row = params.copy()
        row['CONVERGED'] = run_result.converged
        row['PROBLEM'] = run_result.problem
        row['ITERATIONS'] = run_result.iterations
        row['TIME'] = run_result.cpu_time_s

        all_results.append(row)

        run_result = run_one(problems['quad_1000_10'], AlgoSpec(label=label + '_QUAD1000_10', algo_name=method_name, options=options))

        row = params.copy()
        row['CONVERGED'] = run_result.converged
        row['PROBLEM'] = run_result.problem
        row['ITERATIONS'] = run_result.iterations
        row['TIME'] = run_result.cpu_time_s

        all_results.append(row)

        run_result = run_one(problems['quad_1000_1000'], AlgoSpec(label=label + '_QUAD1000_1000', algo_name=method_name, options=options))

        row = params.copy()
        row['CONVERGED'] = run_result.converged
        row['PROBLEM'] = run_result.problem
        row['ITERATIONS'] = run_result.iterations
        row['TIME'] = run_result.cpu_time_s

        all_results.append(row)

        run_result = run_one(problems['quartic_1'], AlgoSpec(label=label + '_QUARTIC1', algo_name=method_name, options=options))

        row = params.copy()
        row['CONVERGED'] = run_result.converged
        row['PROBLEM'] = run_result.problem
        row['ITERATIONS'] = run_result.iterations
        row['TIME'] = run_result.cpu_time_s

        all_results.append(row)

        run_result = run_one(problems['quartic_2'], AlgoSpec(label=label + '_QUARTIC2', algo_name=method_name, options=options))

        row = params.copy()
        row['CONVERGED'] = run_result.converged
        row['PROBLEM'] = run_result.problem
        row['ITERATIONS'] = run_result.iterations
        row['TIME'] = run_result.cpu_time_s

        all_results.append(row)

        

    df = pd.DataFrame(all_results)
    df.to_excel(label + "_ALL_PROBLEMS.xlsx", index=False)
        



if __name__ == "__main__":
    methods = build_methods()
    gridsearch(method_name=methods['BFGS_Wolfe'][0], options=methods['BFGS_Wolfe'][1])