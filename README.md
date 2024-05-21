# nuance
Efficient detection of planets transiting quiet or active stars

<p align="center">
    <img src="docs/_static/illu_readme.png" height="350" style="margin:50px">
</p>

*nuance* uses linear models and Gaussian processes (using the [JAX](https://github.com/google/jax)-based [tinygp](https://github.com/dfm/tinygp)) to simultaneously **search for planetary transits while modeling correlated noises** (e.g. stellar variability) in a tractable way. See [the paper](https://arxiv.org/abs/2402.06835) for more details.

When to use *nuance*?
- To detect single or periodic transits
- When correlated noises are present in the data (e.g. stellar variability or instrumental systematics)
- For space-based or sparse ground-based observations
- To effectively find transits in light curves from multiple instruments
- To use GPUs for fast transit searches

Documentation at [nuance.readthedocs.io](https://nuance.readthedocs.io)

## Example

```python
from nuance import Nuance, utils
import numpy as np

(time, flux, error), X, gp = utils.simulated()

nu = Nuance(time, flux, gp=gp, X=X)

# linear search
epochs = time.copy()
durations = np.linspace(0.01, 0.2, 15)
nu.linear_search(epochs, durations)

# periodic search
periods = np.linspace(0.3, 5, 2000)
search = nu.periodic_search(periods)

t0, D, P = search.best
```

## Installation

`nuance` is written for python 3 and can be installed using pip

```shell
pip install nuance
```

or from sources
  
```shell
git clone https://github.com/lgrcia/nuance
cd nuance
pip install -e .
```
