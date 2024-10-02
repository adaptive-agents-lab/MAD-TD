import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp


@dataclass
class CheckpointHandler:
    checkpoint_dir: str

    def __post_init__(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.options = ocp.CheckpointManagerOptions(
            max_to_keep=3, save_interval_steps=1
        )
        path = os.path.abspath(self.checkpoint_dir)
        print(f"Checkpointer path: {path}")
        self.checkpoint_manager = ocp.CheckpointManager(path, options=self.options)

    def checkpoint_params(self, model, step: int):
        self.checkpoint_manager.save(step, args=ocp.args.StandardSave(model))
        self.checkpoint_manager.wait_until_finished()

    def restore_params(self, models, path: str):
        state = jax.tree_map(jnp.zeros_like, models)
        abstract_train_state = jax.tree_map(ocp.utils.to_shape_dtype_struct, state)
        step = self.checkpoint_manager.latest_step()  # step = 4
        return self.checkpoint_manager.restore(
            step, args=ocp.args.StandardRestore(abstract_train_state)
        )
