import numpy as np

from .name2idx import C
from .observable import Observable
from .ode import param_values


class OptimizationProblem(Observable):
    def __init__(self, search_bounds):
        super(OptimizationProblem, self).__init__()
        self.search_bounds = search_bounds

    @staticmethod
    def _compute_objval_rss(sim_data, exp_data):
        """Return Residual Sum of Squares"""
        return np.dot((sim_data - exp_data), (sim_data - exp_data))

    def update(self, indiv):
        x = param_values()

        for i, param_name in enumerate(self.search_bounds.keys()):
            x[eval(f"C.{param_name}")] = indiv[i]

        # constraints
        x[C.V6] = x[C.V5]
        x[C.Km6] = x[C.Km5]
        x[C.KimpDUSP] = x[C.KimDUSP]
        x[C.KexpDUSP] = x[C.KexDUSP]
        x[C.KimpcFOS] = x[C.KimFOS]
        x[C.KexpcFOS] = x[C.KexFOS]
        x[C.p52] = x[C.p47]
        x[C.m52] = x[C.m47]
        x[C.p53] = x[C.p48]
        x[C.p54] = x[C.p49]
        x[C.m54] = x[C.m49]
        x[C.p55] = x[C.p50]
        x[C.p56] = x[C.p51]
        x[C.m56] = x[C.m51]

        return x

    def get_fval(self, indiv: np.ndarray) -> float:
        x = self.update(indiv)
        self.set_data()

        if self.simulate(x) is None:
            error = np.zeros(2 * len(self.obs_names))
            for i, observable in enumerate(self.obs_names):
                norm_max = np.max(self.simulations[self.obs_names.index(observable)])
                timepoint = self.get_timepoint(observable)
                error[i] = self._compute_objval_rss(
                    self.simulations[self.obs_names.index(observable), self.conditions.index("EGF"), timepoint] / norm_max,
                    self.experiments[self.obs_names.index(observable)]["EGF"],
                )
                error[i + 1] = self._compute_objval_rss(
                    self.simulations[self.obs_names.index(observable), self.conditions.index("HRG"), timepoint] / norm_max,
                    self.experiments[self.obs_names.index(observable)]["HRG"],
                )
            return np.sum(error)  # < 1e12
        else:
            return 1e12
