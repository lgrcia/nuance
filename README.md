> Work in progress ...

# nuance

<p align="center" style="margin-top:20px">
    <img src="docs/_static/nuance.svg" height="250">
</p>

<p align="center">
  A Python package to detect exoplanetary transits <br>in the presence of stellar variability and correlated noises
  <br>
  <p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgray.svg?style=flat" alt="license"/></a>
    <a href="https://nuance.readthedocs.io"><img src="https://img.shields.io/badge/ReadThe-Doc-blue.svg?style=flat" alt="license"/></a>
  </p>
</p>

*nuance* uses linear models and gaussian processes (using the [JAX](https://github.com/google/jax)-based [tinygp](https://github.com/dfm/tinygp)) to simultaneously **search for planetary transits while modeling correlated noises** (e.g. stellar variability) in a tractable way.

When to use *nuance*?
- To detect single or periodic transits
- When correlated noises are present in the data (e.g. stellar variability or instrumental systematics)
- For space-based or sparse ground-based observations
- To effectively find transits in light curves from multiple instruments

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
