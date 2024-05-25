## Hardware acceleration

This reference describes how the linear search is performed depending on the available devices, and how to take advantage of the parallelization capabilities of JAX.

## The linear search

The goal of the linear search is simply to compute 3 quantities for a given epoch `t0` and duration `D`, namely, a single evaluation takes the form

```python
import jax

@jax.jit
def solve(t0, D):
    m = model(time, t0, D)
    ll, w, v = nu._solve(m)
    return w[-1], v[-1, -1], ll
```

where `nu` is a `Nuance` object. For a set of epochs `t0s` and durations `Ds`, the quantities w, v, and ll are computed and saved in 2D grids associated to each pair of (`t0`, `D`). The mapping strategy of nuance consists in choosing how to distribute this computation on different devices.

## Batching

As the trial epochs array is often large (>1000 epochs), nuance separates the `t0s` into batches, so that a mapped version of `solve` can be called on its first argument. Because the trial duration array `Ds` is often shorter (usually < 20 durations), `solve`is then directly mapped on its second argument.

If batches of `BATCH_SIZE` are stored in `batched_t0s`, this is how nuance runs the search:

```python

solve_t0s = map_t0(solve, in_axes=(0, None))
solve_t0s_Ds = map_D(solve_t0s, in_axes=(None, 0))

for _t0s in batched_t0s:
    results.append(solve_t0s_Ds(_t0s, Ds))
```

`map_t0` and `map_D` being the different jax mapping functions (e.g. `vmap` or `pmap`)

## CPUs vs. GPUs
If CPUs are used, `BATCH_SIZE` is chosen as the number of available devices and `map_t0=jax.pmap`, else if 
a GPU is present, `BATCH_SIZE=1000` and `map_t0=jax.vmap`. In both cases `map_D=jax.vmap`.

Of course this strategy is only used as a sensitive default and one can take advantage of their particular architecture to parallelize the evaluation of the `solve` function.