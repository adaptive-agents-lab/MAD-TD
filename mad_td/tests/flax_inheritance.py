from typing import Callable
import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from tqdm import tqdm


class MLP(nn.Module):
    output_dim: int
    act: Callable[[jax.Array], jax.Array]

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.output_dim)(x)
        x = self.act(x)
        return x


class MLP2(MLP):
    act: Callable[[jax.Array], jax.Array] = jnp.tanh


if __name__ == "__main__":

    def create_net(key) -> TrainState:
        net = MLP2(output_dim=2)
        params = net.init(
            key,
            jnp.empty([32, 16]),
        )
        tx = optax.sgd(0.001)
        train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
        return train_state

    train_states = create_net(jax.random.PRNGKey(0))

    def simple_loss(ts, x):
        y = ts.apply_fn(ts.params, x)
        return jnp.mean(y**2)

    jitted_loss = jax.grad(simple_loss, argnums=0, allow_int=True)
    print(train_states)
    for i in tqdm(range(1000)):
        l = jitted_loss(train_states, jnp.ones((32, 16)))
        train_state = train_states.apply_gradients(grads=l.params)
    print(l)
