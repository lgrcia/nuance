# *nuance*

Efficient detection of planets transiting active stars

---

<div style="margin: 100px"></div>

```{image} _static/illu.png
:height: 200px
:align: center
```
<div style="margin: 100px"></div>

*nuance* uses linear models and gaussian processes (using [JAX](https://github.com/google/jax)-based [tinygp](https://github.com/dfm/tinygp)) to simultaneously **search for planetary transits while modeling correlated noises** (e.g. stellar variability) in a tractable way. 

Paper released [on arxiv](https://arxiv.org/abs/2402.06835).

```{toctree}
:maxdepth: 1
:caption: Get started

markdown/install
notebooks/single.ipynb
notebooks/periodic.ipynb
notebooks/multi.ipynb
notebooks/combined.ipynb
```


```{toctree}
:maxdepth: 1
:caption: Tutorials

notebooks/tutorials/ground_based.ipynb
notebooks/tutorials/tess_search.ipynb
notebooks/tutorials/exocomet.ipynb
```

```{toctree}
:maxdepth: 1
:caption: Reference

markdown/how
markdown/API
```
