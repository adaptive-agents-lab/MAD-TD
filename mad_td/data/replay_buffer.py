import os
from dataclasses import dataclass
from typing import Sequence


import jax
import jax.numpy as jnp
import numpy as np

from mad_td.rl_types import Dataset, RLBatch
from mad_td.data.env import Env


@dataclass
class ReplayBuffer(Dataset):
    state_shape: Sequence[int]
    action_shape: Sequence[int]
    env: Env
    num_seeds: int
    capacity: int
    rollout_length: int
    is_img_obs: bool = False
    is_discrete_action: bool = False

    def __post_init__(
        self,
    ):
        state_type = np.uint8 if self.is_img_obs else np.float32
        action_type = np.uint16 if self.is_discrete_action else np.float32
        # explicitly setting the arrays to 0 forces allocation and prevents later failures due to memory end
        self.states = np.empty(
            (self.num_seeds, self.capacity, *self.state_shape), dtype=state_type
        )
        self.states.fill(0.0)
        self.actions = np.empty(
            (self.num_seeds, self.capacity, *self.action_shape), dtype=action_type
        )
        self.actions.fill(0.0)
        self.rewards = np.empty(
            (self.num_seeds, self.capacity, 1),
            dtype=np.float32,
        )
        self.rewards.fill(0.0)
        self.next_states = np.empty(
            (self.num_seeds, self.capacity, *self.state_shape), dtype=state_type
        )
        self.next_states.fill(1.0)
        self.masks = np.empty(
            (self.num_seeds, self.capacity, 1),
            dtype=np.float32,
        )
        self.masks.fill(0.0)
        self.filled = 0
        self.insert_index = 0
        self.saved_index = 0
        self.saved_counter = 0


    def insert(
        self,
        state: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        done: jax.Array,
        next_state: jax.Array,
    ):
        state = np.array(state)
        action = np.array(action)
        next_state = np.array(next_state)
        reward = np.array(reward)
        done = np.array(done)

        self.states[:, self.insert_index] = state
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index, 0] = reward
        self.masks[:, self.insert_index, 0] = np.logical_not(done)
        self.next_states[:, self.insert_index] = next_state

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.filled = min(self.filled + 1, self.capacity)

    def __len__(self):
        return max(0, self.filled - self.rollout_length)

    def sample(self, batch_size: int, key: jax.Array, batches: int) -> RLBatch:
        idxs = np.random.randint(
            0,
            max(1, len(self)),
            size=(
                batches,
                batch_size,
            ),
        )
        expanded_idxs = idxs[..., None] + np.arange(self.rollout_length)
        states = self.states[:, idxs]
        actions = self.actions[:, expanded_idxs]
        rewards = self.rewards[:, expanded_idxs]
        next_states = self.next_states[:, expanded_idxs]
        masks = self.masks[:, expanded_idxs]
        masks = np.cumprod(masks, axis=-2)

        return RLBatch(
            state=states,
            action=actions,
            reward=rewards,
            next_state=next_states,
            idxs=np.ones((self.num_seeds,1,1)) * idxs,
            mask=masks,
        )

    def get_dummy_batch(self, num_samples=None, batch_size=0) -> RLBatch:
        if num_samples is None:
            return RLBatch(
                state=jnp.array(self.states[:, 0]),
                action=jnp.array(self.actions[:, 0]),
                reward=jnp.array(self.rewards[:, 0]),
                next_state=jnp.array(self.next_states[:, 0]),
                idxs=jnp.zeros((self.num_seeds, 1)),
                mask=jnp.array(self.masks[:, 0]),
            )
        else:
            key = jax.random.PRNGKey(0)
            return self.sample(batch_size, key, num_samples)

    def batch_iterator(self, batch_size: int, key: jax.Array):
        sample_idx = 0
        while sample_idx < len(self):
            idxs = np.array(
                list(range(sample_idx, min(sample_idx + batch_size, len(self))))
            )
            expanded_idxs = idxs[..., None] + np.arange(self.rollout_length)
            states = self.states[:, idxs]
            actions = self.actions[:, expanded_idxs]
            rewards = self.rewards[:, expanded_idxs]
            next_states = self.next_states[:, expanded_idxs]
            masks = self.masks[:, expanded_idxs]
            masks = np.cumprod(masks, axis=-2)
            sample_idx += batch_size

            yield RLBatch(
                state=states,
                action=actions,
                reward=rewards,
                next_state=next_states,
                idxs=idxs,
                mask=masks,
            )

    def save(self, path):
        path = os.path.join(path, "buffer")
        if not os.path.exists(path):
            os.makedirs(path)
        outfile = os.path.join(path, f"replay_buffer_{self.saved_index:07d}.npz")
        tmp_outfile = "tmp.npz"
        np.savez(
            tmp_outfile,
            s=self.states[:, self.saved_counter : self.insert_index],
            a=self.actions[:, self.saved_counter : self.insert_index],
            r=self.rewards[:, self.saved_counter : self.insert_index],
            sn=self.next_states[:, self.saved_counter : self.insert_index],
            m=self.masks[:, self.saved_counter : self.insert_index],
            ii=self.insert_index,
            size=self.filled,
            counter=self.saved_counter,
        )
        os.replace(tmp_outfile, outfile)
        print(f"Saved from {self.saved_counter} to {self.insert_index} in {outfile}")
        self.saved_index += 1
        self.saved_counter = self.insert_index

    def load(self, path):
        path = os.path.join(path, "buffer")
        files = os.listdir(path)
        files.sort()
        for i, infile in enumerate(files):
            infile = os.path.join(path, infile)
            data = np.load(infile)
            start = int(data["counter"])
            end = int(data["ii"])
            self.states[:, start:end] = data["s"]
            self.actions[:, start:end] = data["a"]
            self.rewards[:, start:end] = data["r"]
            self.next_states[:, start:end] = data["sn"]
            self.masks[:, start:end] = data["m"]
            print(f"Reading file {i}, filling {start} to {end}")
        self.insert_index = int(data["ii"])
        self.filled = int(data["size"])
        self.saved_counter = self.filled
        self.saved_index = len(files) + 1
        print(f"Resuming from {self.saved_counter}")
