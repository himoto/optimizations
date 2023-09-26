import os
import shutil

NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS

import numpy as np

from benchmark_function.benchmark_model import BenchmarkModel
from src import optimizations

model = BenchmarkModel()


def fun_optim(_, params: np.ndarray) -> float:
    """Objective function to be minimized."""
    assert params.ndim == 1
    objval = model.objective(params)
    return objval


def main():
    # cleanup
    if os.path.exists(output_file := "pybnf_output"):
        shutil.rmtree(output_file)
    x_optim = np.array([0])
    y_optim = np.array([0])

    # set up configuration
    params = [optimizations.UniformParam(var_type="loguniform_var", lower_bound=lb, upper_bound=ub) for (lb, ub) in model.search_bounds.values()]
    param_config = optimizations.ParamConfig(params=params)
    algorithm_config = optimizations.AlgConfig_ScatterSearch()
    general_config = optimizations.GeneralConfig(
        param_config=param_config,
        algorithm_config=algorithm_config,
        population_size=20,
        max_iterations=30,
        verbosity=0,
        objfunc="sos",
        min_objective=0,
    )
    res = optimizations.run_simple_optimization(
        func=fun_optim,
        inputs=x_optim,
        outputs=y_optim,
        general_config=general_config,
    )
    print(res)
    model.show_optimization_results(xopt=res.x, fname="fitting_result")


if __name__ == "__main__":
    main()
