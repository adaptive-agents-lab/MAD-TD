import jax
import jax.numpy as jnp

import flax.linen as nn
import optax

from mad_td.utils.jax import ExpandedTrainState


class RunningMeanNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        r_mean = self.variable("variables", "running_mean", lambda: jnp.ones((1,)))
        mean = jnp.mean(x**2) + 1e-8
        r_mean.value = 0.99 * r_mean.value + 0.01 * mean
        return x / r_mean.value


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = RunningMeanNorm()(x)
        x = nn.Dense(1)(x)
        return x


def update_step(train_state: ExpandedTrainState, x, y):
    p = train_state.params
    v = train_state.variables

    def loss_fn(parameters):
        y_pred, updated_values = train_state.apply_fn(
            {"params": parameters, "variables": v}, x, mutable=["variables"]
        )
        return jnp.mean((y - y_pred) ** 2), updated_values

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, variables), grad = grad_fn(p)
    train_state = train_state.apply_gradients(
        grads=grad, variables=variables["variables"]
    )
    return loss, train_state


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1000, 10))
    y = jax.random.normal(key, (1000, 1))

    model = MLP()
    params = model.init(key, x)
    params, v = params.pop("variables")
    train_state = ExpandedTrainState.create(
        apply_fn=MLP().apply,
        params=params["params"],
        variables=v,
        tx=optax.adam(1e-3),
    )

    for _ in range(10000):
        loss, train_state = update_step(train_state, x, y)
        print(train_state.variables["RunningMeanNorm_0"]["running_mean"])
        print(loss)
