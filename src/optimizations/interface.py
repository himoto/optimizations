"""Module providing a usable interface to run optimizations."""

import os
import shutil

import numpy as np
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
    # TODO do not start from default_config, instead start from empty dict

    # custom parameters
    # TODO document custom parameters in CustomConfiguration!!
    param_dict["models"] = "np"
    param_dict["numpy_model"] = True
    param_dict["_optimization"] = [
        "_data",
    ]
    param_dict["_custom_data"] = data
    param_dict["_custom_func"] = func

    # param_dict["fit_type"] = "de"
    # test_param_dict["fit_type"] = "ss"
    param_dict["population_size"] = 25
    param_dict["max_iterations"] = 30
    param_dict["objfunc"] = "sos"

    # TODO set output dir to temporary dir and clean up afterwards

    # TODO investigate how to handle unbound parameters
    # TODO consider whether we only want uniform_var parameters
    # parameters
    for i in range(n_params):
        param_dict[("uniform_var", f"v{i:0{10}d}__FREE")] = [0.0, 10.0, True]
        # param_dict[("uniform_var", f"v{i:0{10}d}__FREE")] = [-10000000, 10000000, True]

    # for testing, disable dusk parallelization
    # test_param_dict["parallelize_models"] = 0

    ###########################################################################

    # for k, v in config.config.items():
    # print(k, v)

    # TODO match algorithm type
    param_dict["fit_type"] = "de"
    config = CustomConfiguration(param_dict)
    # alg = pybnf.algorithms.DifferentialEvolution(config)

    param_dict["fit_type"] = "ade"
    config = CustomConfiguration(param_dict)
    # alg = pybnf.algorithms.AsynchronousDifferentialEvolution(config)

    param_dict["fit_type"] = "ss"
    config = CustomConfiguration(param_dict)
    # alg = pybnf.algorithms.ScatterSearch(config)

    param_dict["fit_type"] = "pso"
    config = CustomConfiguration(param_dict)
    # alg = pybnf.algorithms.ParticleSwarm(config)

    param_dict["fit_type"] = "mh"
    param_dict["max_iterations"] = 300
    param_dict["burn_in"] = 100
    config = CustomConfiguration(param_dict)
    # alg = pybnf.algorithms.BasicBayesMCMCAlgorithm(config)
    # TODO mh also produces an output file called Results/credible95_final.txt, which I
    # assume contains 95% confidence intervals for estimated parameters
    # -> add to OptimizeResult output

    param_dict["fit_type"] = "pt"
    param_dict["max_iterations"] = 300
    param_dict["burn_in"] = 100
    config = CustomConfiguration(param_dict)
    # alg = pybnf.algorithms.BasicBayesMCMCAlgorithm(config)

    param_dict["fit_type"] = "am"
    param_dict["max_iterations"] = 500
    param_dict["burn_in"] = 100
    param_dict["adaptive"] = 100  # max_iterations must be at least 2 x (burn_in + adaptive)
    config = CustomConfiguration(param_dict)
    # alg = pybnf.algorithms.Adaptive_MCMC(config)

    param_dict["fit_type"] = "sa"
    param_dict["max_iterations"] = 300
    param_dict["burn_in"] = 100
    config = CustomConfiguration(param_dict)
    alg = pybnf.algorithms.BasicBayesMCMCAlgorithm(config, sa=True)

    param_dict['fit_type'] = 'sim'
    param_dict["max_iterations"] = 100
    # config = CustomConfiguration(param_dict)
    # alg = pybnf.algorithms.SimplexAlgorithm(config)
    # TODO Simplex does not work straightforward, since it requires initial values
    # instead of parameter ranges -> think how to implement this

    # TODO do not provide an entry point to the DREAM algorithm, seems to be experimental atm
    # param_dict['fit_type'] = 'dream'
    # param_dict["max_iterations"] = 100
    # config = CustomConfiguration(param_dict)
    # alg = pybnf.algorithms.DreamAlgorithm(config)

    # # IMPORTANT: it is necessary to create pybnf_output/Simulations dir!
    # # IMPORTANT: pybnf_output/Results seems also important!
    os.makedirs(os.path.join(config.config["output_dir"], "Simulations"), exist_ok=True)
    os.makedirs(os.path.join(config.config["output_dir"], "Results"), exist_ok=True)

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
