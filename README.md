[![Lifecycle:Experimental](https://img.shields.io/badge/Lifecycle-Experimental-339999)](https://github.com/PhilippJunk/optimizations)

# optimizations

This python package is a wrapper around the `pybnf` python package, which allows to run the sophisticated optimization algorithms in that package with any arbitrary python function instead of formulating it as a BNGL or SBML model.

-----

**Table of Contents**

- [Installation](#installation)
- [Example](#example)
- [License](#license)

## Installation

This package requires `python >= 3.10`. It has been tested against `pybnf == 1.2.2`.

## Example

This is an example fitting a 2nd degree polynomial to some data. This example is 
directly taken from the pyBNF demo example.

```python
import numpy as np
import optimizations

# define function
def parabola(data, params):
    assert params.size == 3

    a, b, c = params[0], params[1], params[2]

    return a * data**2 + b * data + c


# data
x = np.array([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,])
y = np.array([43.0, 34.5, 27.0, 20.5, 15.0, 10.5, 7.0, 4.5, 3.0, 2.5, 3.0, 4.5, 7.0, 10.5, 15.0, 20.5, 27.0, 34.5, 43.0, 52.5, 63.0,])

# set up configuration objects
param_config = optimizations.all_equal_bounds(
    n_params=3,
    var_type="uniform_var",
    lower_bound=-10.0,
    upper_bound=10.0,
)
algorithm_config = optimizations.AlgConfig_DifferentialEvolution()
general_config = optimizations.GeneralConfig(
    param_config=param_config,
    algorithm_config=algorithm_config,
    population_size=50,
    max_iterations=100,
    verbosity=0,
    objfunc="sos",
)

# run optimization
res = optimizations.run_simple_optimization(
    func=parabola,
    inputs=x,
    outputs=y,
    general_config=general_config,
)
```

## License

`optimizations` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
