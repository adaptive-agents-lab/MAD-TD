import jax
import jax.numpy as jnp

from mad_td.utils import jax as jax_utils  # import hl_gauss, two_hot, symlog, symexp
from mad_td.nn import critic


def decode(cat, num_bins, vmin, vmax):
    support = jax.numpy.linspace(vmin, vmax, num_bins)
    x = cat.dot(support)
    # return symexp(x)
    return x


def test_initalize():
    def l(i):
        prng_key = jax.random.PRNGKey(i)
        model = critic.BinnedTDMPC2Critic([512, 512], 151, -200, 200)

        input1 = jax.random.normal(prng_key, shape=(1, 256))
        input2 = jax.random.normal(prng_key, shape=(1, 15))
        params = model.init(prng_key, input1, input2)
        x1 = jnp.minimum(*model.apply(params, input1, input2))

        model = critic.ViperCritic([512, 512])

        input1 = jax.random.normal(prng_key, shape=(1, 256))
        input2 = jax.random.normal(prng_key, shape=(1, 15))
        params = model.init(prng_key, input1, input2)
        x2 = jnp.minimum(*model.apply(params, input1, input2))
        return x1, x2

    keys = jnp.array(range(1000))

    gauss_inits, viper_inits = jax.vmap(jax.jit(l))(keys)

    print(jnp.mean(gauss_inits), jnp.mean(viper_inits))
    print(jnp.std(gauss_inits), jnp.std(viper_inits))


def test():
    num_bins = 132
    x = jnp.array([-100, -50, 14, 64.45, -5, 0, 5, 50, 100])
    cat = jax.vmap(jax_utils.hl_gauss, in_axes=[0, None, None, None])(
        x, num_bins, -151, 151
    )
    x_rec = jax.vmap(decode, in_axes=[0, None, None, None])(
        cat, num_bins, -151, 151
    ).squeeze()
    print(jnp.mean((x_rec - x) ** 2))

    x = jnp.array([-100, -50, 14, 64.45, -5, 0, 5, 50, 100])
    cat = jax.vmap(jax_utils.two_hot, in_axes=[0, None, None, None])(
        x, num_bins, -151, 151
    )
    x_rec = jax.vmap(decode, in_axes=[0, None, None, None])(
        cat, num_bins, -151, 151
    ).squeeze()
    print(jnp.mean((x_rec - x) ** 2))


if __name__ == "__main__":
    test()
    test_initalize()
