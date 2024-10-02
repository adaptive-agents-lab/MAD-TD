import abc
from typing import Tuple, Union

import jax
import flax.linen as nn
from flax import struct


class AbstractCritic(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, states: jax.Array, actions: jax.Array
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        pass


class AbstractActor(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(self, states: jax.Array, actions: jax.Array) -> jax.Array:
        pass


class AbstractEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(self, states: jax.Array) -> jax.Array:
        pass


class AbstractLatentModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, states: jax.Array, actions: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        pass


class AbstractBatch(abc.ABC):
    pass


@struct.dataclass
class RLBatch(AbstractBatch):
    state: jax.Array
    action: jax.Array
    reward: jax.Array
    next_state: jax.Array
    mask: jax.Array
    idxs: jax.Array
    physics_state: jax.Array | None = None


def select_idx_batch(batch: RLBatch, idx: int) -> RLBatch:
    if batch.physics_state is not None:
        return RLBatch(
            state=batch.state[:, idx],
            action=batch.action[:, idx],
            reward=batch.reward[:, idx],
            next_state=batch.next_state[:, idx],
            mask=batch.mask[:, idx],
            idxs=batch.idxs[:, idx],
            physics_state=batch.physics_state[:, idx],
        )
    else:
        return RLBatch(
            state=batch.state[:, idx],
            action=batch.action[:, idx],
            reward=batch.reward[:, idx],
            next_state=batch.next_state[:, idx],
            mask=batch.mask[:, idx],
            idxs=batch.idxs[:, idx],
            physics_state=None,
        )


def remove_first_batch(batch: RLBatch) -> RLBatch:
    if batch.physics_state is not None:
        return RLBatch(
            state=batch.state[:, 1:],
            action=batch.action[:, 1:],
            reward=batch.reward[:, 1:],
            next_state=batch.next_state[:, 1:],
            mask=batch.mask[:, 1:],
            idxs=batch.idxs[:, 1:],
            physics_state=batch.physics_state[:, 1:],
        )
    else:
        return RLBatch(
            state=batch.state[:, 1:],
            action=batch.action[:, 1:],
            reward=batch.reward[:, 1:],
            next_state=batch.next_state[:, 1:],
            idxs=batch.idxs[:, 1:],
            mask=batch.mask[:, 1:],
        )


def truncate_batch(batch: RLBatch, length: int) -> RLBatch:
    if batch.physics_state is not None:
        return RLBatch(
            state=batch.state[:, :length],
            action=batch.action[:, :length],
            reward=batch.reward[:, :length],
            next_state=batch.next_state[:, :length],
            mask=batch.mask[:, :length],
            idxs=batch.idxs[:, :length],
            physics_state=batch.physics_state[:, :length],
        )
    else:
        return RLBatch(
            state=batch.state,
            action=batch.action[:, :length],
            reward=batch.reward[:, :length],
            next_state=batch.next_state[:, :length],
            idxs=batch.idxs[:, :length],
            mask=batch.mask[:, :length],
        )


def get_batch_seed_shape(batch: RLBatch):
    return jax.tree_map(lambda x: 0 if x is not None else None, batch)


class Dataset(abc.ABC):
    @abc.abstractmethod
    def sample(self, batch_size: int, key: int, batches: int) -> AbstractBatch:
        pass

    @abc.abstractmethod
    def load(self, path: str):
        pass

    @abc.abstractmethod
    def save(self, path: str):
        pass
