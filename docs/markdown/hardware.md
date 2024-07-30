# Hardware acceleration

When running the *linear search*, nuance exploits the parallelization capabilities of JAX by using a default mapping strategy depending on the available devices.

## Solving for `(t0, D)`
 
To solve a particular model (like a transit) with a given epoch `t0` and duration `D`, we define the function (output of `core.solve`)

```python
import jax

@jax.jit
def solve(t0, D, period=None):
    m = model(time, t0, D, period=period)
    ll, w, v = solve_m(m)
    return jnp.array([w[-1], v[-1, -1], ll])
```

where `model` is the [template model](../notebooks/templates.ipynb). This function returns

- `w[-1]` the template model depth
- `v[-1, -1]` the variance of the template model depth
- `ll` the log-likelihood of the data to the model

## Batching over `(t0s, Ds)`
The goal of the linear search is then to call `solve` for a grid of of epochs `t0s` and durations `Ds`. As `t0s` is usually very large compared to `Ds` (~1000 vs. ~10), the default strategy is to batch the `t0s`:

```python
# we pad to have fixed size batches
t0s_padded = np.pad(t0s, [0, batch_size - (len(t0s) % batch_size) % batch_size])
t0s_batches = np.reshape(
    t0s_padded, (len(t0s_padded) // batch_size, batch_size)
)
```

## JAX mapping

In order to solve a given batch in an optimal way, the `batch_size` can be set depending on the devices available (see the `linear_search` documentation):

- If multiple **CPUs** are available, the `batch_size` is chosen as the number of devices (`jax.device_count()`) and we can solve a given batch using

    ```python
    solve_batch = jax.pmap(jax.vmap(solve, in_axes=(None, 0)), in_axes=(0, None))
    ```

    where each batch is `jax.pmap`ed over all available CPUs along the `t0s` axis. 

- If a **GPU** is available, the `batch_size` can be larger and the batch is `jax.vmap`ed along `t0s`

    ```python
    solve_batch = jax.vmap(jax.vmap(solve, in_axes=(None, 0)), in_axes=(0, None))
    ```

Then, the linear search consists in iterating over `t0s_batches`:

```python
results = []

for t0_batch in t0s_batches:
    results.append(solve_batch(t0_batch, Ds))
```

```{note}
Of course, one familiar with JAX can use their own mapping strategy to evaluate `solve` over a grid of epochs `t0s` and durations `Ds`. For these users, the implementation of the `linear_search` method is a good place to start.
```
