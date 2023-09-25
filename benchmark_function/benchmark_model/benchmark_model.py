import warnings
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from .name2idx import C
from .ode import param_values
from .problem import OptimizationProblem


@dataclass
class BenchmarkModel(OptimizationProblem):
    """
    Benchmark model and function for optimization.
    For more details about the model, please refer to the following paper:
    * Nakakuki, T. et al. Ligand-specific c-Fos expression emerges from the spatiotemporal control of ErbB network dynamics. Cell 141, 884–896 (2010). https://doi.org/10.1016/j.cell.2010.03.054

    Parameters
    ----------
    search_bounds : dict
        Keys: Parameter name.
        Values: Lower and upper bounds for each parameter.
    """

    search_bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "V1": (7.33e-2, 6.60e-01),
            "Km1": (1.83e2, 8.50e2),
            "V5": (6.48e-3, 7.20e1),
            "Km5": (6.00e-1, 1.60e04),
            "V10": (np.exp(-10), np.exp(10)),
            "Km10": (np.exp(-10), np.exp(10)),
            "n10": (1.00, 4.00),
            "p11": (8.30e-13, 1.44e-2),
            "p12": (8.00e-8, 5.17e-2),
            "p13": (1.38e-7, 4.84e-1),
            "V14": (4.77e-3, 4.77e1),
            "Km14": (2.00e2, 2.00e6),
            "V15": (np.exp(-10), np.exp(10)),
            "Km15": (np.exp(-10), np.exp(10)),
            "KimDUSP": (2.20e-4, 5.50e-1),
            "KexDUSP": (2.60e-4, 6.50e-1),
            "V20": (4.77e-3, 4.77e1),
            "Km20": (2.00e2, 2.00e6),
            "V21": (np.exp(-10), np.exp(10)),
            "Km21": (np.exp(-10), np.exp(10)),
            "V24": (4.77e-2, 4.77e0),
            "Km24": (2.00e3, 2.00e5),
            "V25": (np.exp(-10), np.exp(10)),
            "Km25": (np.exp(-10), np.exp(10)),
            "KimRSK": (2.20e-4, 5.50e-1),
            "KexRSK": (2.60e-4, 6.50e-1),
            "V27": (np.exp(-10), np.exp(10)),
            "Km27": (1.00e2, 1.00e4),
            "V28": (np.exp(-10), np.exp(10)),
            "Km28": (np.exp(-10), np.exp(10)),
            "V29": (4.77e-2, 4.77e0),
            "Km29": (2.93e3, 2.93e5),
            "V30": (np.exp(-10), np.exp(10)),
            "Km30": (np.exp(-10), np.exp(10)),
            "V31": (np.exp(-10), np.exp(10)),
            "Km31": (np.exp(-10), np.exp(10)),
            "n31": (1.00, 4.00),
            "p32": (8.30e-13, 1.44e-2),
            "p33": (8.00e-8, 5.17e-2),
            "p34": (1.38e-7, 4.84e-1),
            "V35": (4.77e-3, 4.77e1),
            "Km35": (2.00e2, 2.00e6),
            "V36": (np.exp(-10), np.exp(10)),
            "Km36": (1.00e2, 1.00e4),
            "V37": (np.exp(-10), np.exp(10)),
            "Km37": (np.exp(-10), np.exp(10)),
            "KimFOS": (2.20e-4, 5.50e-1),
            "KexFOS": (2.60e-4, 6.50e-1),
            "V42": (4.77e-3, 4.77e1),
            "Km42": (2.00e2, 2.00e6),
            "V43": (np.exp(-10), np.exp(10)),
            "Km43": (1.00e2, 1.00e4),
            "V44": (np.exp(-10), np.exp(10)),
            "Km44": (np.exp(-10), np.exp(10)),
            "p47": (1.45e-4, 1.45e0),
            "m47": (6.00e-3, 6.00e1),
            "p48": (2.70e-3, 2.70e1),
            "p49": (5.00e-5, 5.00e-1),
            "m49": (5.00e-3, 5.00e1),
            "p50": (3.00e-3, 3.00e1),
            "p51": (np.exp(-10), np.exp(10)),
            "m51": (np.exp(-10), np.exp(10)),
            "V57": (np.exp(-10), np.exp(10)),
            "Km57": (np.exp(-10), np.exp(10)),
            "n57": (1.00, 4.00),
            "p58": (8.30e-13, 1.44e-2),
            "p59": (8.00e-8, 5.17e-2),
            "p60": (1.38e-7, 4.84e-1),
            "p61": (np.exp(-10), np.exp(10)),
            "KimF": (2.20e-4, 5.50e-1),
            "KexF": (2.60e-4, 6.50e-1),
            "p63": (np.exp(-10), np.exp(10)),
            "KF31": (np.exp(-10), np.exp(10)),
            "nF31": (1.00, 4.00),
            "a": (1.00e2, 5.00e2),
        }
    )

    def __post_init__(self):
        self.search_bounds_log = np.log10(np.abs(list(self.search_bounds.values())))

    def gene2val(self, genes: np.ndarray) -> np.ndarray:
        """
        Convert genes into parameter values.
        """
        param_values = 10 ** (genes * (self.search_bounds_log[:, 1] - self.search_bounds_log[:, 0]) + self.search_bounds_log[:, 0])
        for i, (param_name, (lb, ub)) in enumerate(self.search_bounds.items()):
            if lb > 0 and ub > 0:
                pass
            elif lb < 0 and ub < 0:
                param_values[i] *= -1
            else:
                raise ValueError(f"{param_name}: Sign of lower and upper bounds must be the same.")
        return param_values

    def objective(self, x: np.ndarray):
        """
        Objective function to be minimized.

        Parameters
        ----------
        x : np.ndarray
            Parameter vector.

        Returns
        -------
        objval : float
            Objective function value.
        """
        assert x.ndim == 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            objval = self.get_fval(x)
        return objval

    def objective_log(self, gene: np.ndarray) -> float:
        """
        Objective function to be minimized.

        Parameters
        ----------
        gene : np.ndarray
            Gene vector.

        Returns
        -------
        objval : float
            Objective function value.
        """
        x = self.gene2val(gene)
        return self.objective(x)

    def show_optimization_results(self, *, xopt: np.ndarray, fname: str):
        """
        Visualize the optimization result to assess the performance of an optimizer.
        Lines and dots denote simulations and experimental data, respectively.

        Parameters
        ----------
        xopt : 1D np.ndarray
            Parameter vector (optimized).
        fname : str
            File name.
        """
        x = param_values()
        for i, param_name in enumerate(self.search_bounds.keys()):
            x[eval(f"C.{param_name}")] = xopt[i]
        assert self.simulate(x) is None, "Simulation failed!"
        self.set_data()
        plt.figure(figsize=(12, 6))
        for i, observable in enumerate(self.obs_names):
            plt.subplot(2, 4, i + 1)
            exp_t = self.get_timepoint(observable)
            norm_max = np.max(self.simulations[i])
            plt.plot(self.t, self.simulations[i, 0] / norm_max, "b")
            plt.plot(exp_t, self.experiments[i]["EGF"], "o", color="b")
            plt.plot(self.t, self.simulations[i, 1] / norm_max, "r")
            plt.plot(exp_t, self.experiments[i]["HRG"], "o", color="r")
            plt.xlabel("Time")
            plt.ylabel(observable)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
