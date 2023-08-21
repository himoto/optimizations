"""Module providing a usable interface to run optimizations."""

import os
import shutil

import pandas as pd
import pybnf
import pybnf.cluster
import pybnf.algorithms

from .custom_classes import CustomData, CustomConfiguration

## NOTES
# TODO make a separate dataclass for all possible param_dicts (1 per optimization algorithm)
# TODO make enum with all those dataclasses and match over them to set algo


def run_simple_optimization(func, inputs, outputs, n_params):
    """
    Run simple optimization using pyBNF differential evoluation algorithm.
    """
    data = CustomData.from_x_and_y(inputs, outputs)

    # Create parameter dict
    ###########################################################################
    param_dict = CustomConfiguration.default_config()

    # custom parameters
    # TODO document custom parameters in CustomConfiguration!!
    param_dict["models"] = "np"
    param_dict["numpy_model"] = True
    param_dict["_optimization"] = [
        "_data",
    ]
    param_dict["_custom_data"] = data
    param_dict["_custom_func"] = func

    param_dict["fit_type"] = "de"
    # test_param_dict["fit_type"] = "ss"
    param_dict["population_size"] = 50
    param_dict["max_iterations"] = 1000
    param_dict["objfunc"] = "sos"

    # TODO set output dir to temporary dir and clean up afterwards

    # parameters
    for i in range(n_params):
        param_dict[("uniform_var", f"v{i:0{10}d}__FREE")] = [0.0, 10.0, True]

    # for testing, disable dusk parallelization
    # test_param_dict["parallelize_models"] = 0

    ###########################################################################

    config = CustomConfiguration(param_dict)
    # for k, v in config.config.items():
    # print(k, v)

    # # IMPORTANT: it is necessary to create pybnf_output/Simulations dir!
    # # IMPORTANT: pybnf_output/Results seems also important!
    os.makedirs(os.path.join(config.config["output_dir"], "Simulations"), exist_ok=True)
    os.makedirs(os.path.join(config.config["output_dir"], "Results"), exist_ok=True)

    # TODO match algorithm type
    alg = pybnf.algorithms.DifferentialEvolution(config)

    # TODO check if cluster is actually useful. If not, mock it
    # TODO check arguments for cluster
    cluster = pybnf.cluster.Cluster(config, "test", False, "info")
    alg.run(cluster.client, resume=None, debug=False)

    # load results
    results = pd.read_table(os.path.join(config.config["output_dir"], "Results", "sorted_params_final.txt"))

    # delete dir
    shutil.rmtree(config.config['output_dir'])

    # TODO process results to extract parameters, wrap in scipy.optimize.OptimizeResult
    # TODO catch any errors during optimization, wrap in scipy.optimize.OptimizeResult

    return results
