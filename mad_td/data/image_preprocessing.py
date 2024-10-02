import jax
import jax.numpy as jnp

import jax.scipy.ndimage as jndi


@jax.jit
def random_shift_aug(obs: jax.Array, pad: int, key: jax.Array):
    h, w, c = obs.shape

    arange_x = jnp.arange(0, h, 1)
    arange_x = arange_x[jnp.newaxis].repeat(h, 0)[..., jnp.newaxis]
    arange_y = arange_x.transpose(1, 0, 2)

    offset = jax.random.uniform(key, (2,)) * 4
    arange_x = arange_x + offset[0]
    arange_y = arange_y + offset[1]

    return jndi.map_coordinates(
        obs, [arange_y, arange_x, [[jnp.arange(0, c, 1)]]], order=1, mode="nearest"
    )


def image_normalize(x):
    return (x / (255.0 / 2.0)) - 1.0
