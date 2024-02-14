# *nuance*

Efficient detection of planets transiting active stars

---

<div style="margin: 100px"></div>

```{image} _static/illu.png
:height: 200px
:align: center
```
<div style="margin: 50px"></div>


````{grid} 3
---
padding: 0
margin: 0
gutter: 0
---

```{grid-item-card} 🐤 Where to start?
---
class-card: sd-border-0
shadow: None
---

Read the [Motivation](./notebooks/motivation.ipynb) behind the development of *nuance* and 
check out the basic transit search [Examples](./examples).
```

```{grid-item-card} 📦 Applications
---
class-card: sd-border-0
shadow: None
---

Explore the applications of nuance on realistic datasets, such 
as [TESS](./notebooks/tutorials/tess_search) or [Ground-based](./notebooks/tutorials/ground_based) observations.
```

```{grid-item-card} ⚙️ More details
---
class-card: sd-border-0
shadow: None
---

Read the [*nuance* paper](https://arxiv.org/abs/2402.06835), learn [How the package works]() or study the 
[API](./markdown/API.md).
```

````

*nuance* uses linear models and gaussian processes (using [JAX](https://github.com/google/jax)-based [tinygp](https://github.com/dfm/tinygp)) to simultaneously **search for planetary transits while modeling correlated noises** (e.g. stellar variability) in a tractable way. 


---

```{toctree}
:maxdepth: 1
:caption: Get started

markdown/install
notebooks/motivation.ipynb
notebooks/star.ipynb
notebooks/templates.ipynb
examples.md
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

markdown/how.ipynb
markdown/API
```
