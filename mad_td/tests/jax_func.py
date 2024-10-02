import jax.numpy as jnp
from mad_td.utils.jax import batch_loss_fn

test = jnp.array([[1, 20.0], [2, 3.0]])


def f(x, g):
    return g(x), {"res": g(x)}


batch_fn = batch_loss_fn(f, in_axes=(0, None), out_axes=(0, 0), has_aux=True)
# batch_fn = jax.vmap(f, in_axes=(0, None), out_axes=(0, 0))
print(batch_fn(test, jnp.sum))
