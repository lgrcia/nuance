# *nuance*

Efficient detection of planets transiting active stars

---

<div style="margin: 100px"></div>

```{image} _static/illu.png
:height: 200px
:align: center
```
<div style="margin: 30px"></div>
<style>
.flex {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
      justify-content: space-evenly
}
.max-w-100 {
    max-width: 14rem;
}
</style>

<div class="flex flex-row  gap-5 mb-5">
    <div class="max-w-100">
        <div class="text-center mb-2">
        <img class="p-2" src="_static/box.png"/></div>
        <div class="text-center mb-2">
        <strong>Where to start</strong></div>
        Read the <a href="./notebooks/motivation.html">Motivation</a> behind the development of <i>nuance</i> and 
        check out the basic transit search <a href="./examples.html">Examples</a>.
    </div>
    <div class="max-w-100">
        <div class="text-center mb-2">
        <img class="p-2" src="_static/star.png"/></div>
        <div class="text-center mb-2">
        <strong>Applications</strong></div>
        Explore the applications of nuance on realistic datasets, such 
        as <a href="./notebooks/tutorials/tess_search.html">TESS</a> or <a href="./notebooks/tutorials/ground_based.html">Ground-based</a> observations.
    </div>
    <div class="max-w-100">
        <div class="text-center mb-2">
        <img class="p-2" src="_static/wire.png"/></div>
        <div class="text-center mb-2">
        <strong>The details</strong></div>
        Read the <a href="https://arxiv.org/abs/2402.06835">nuance paper</a> or study the <a href="./markdown/API.html">API</a>
    </div>
</div>



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
