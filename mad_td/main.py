import os
import getpass

import hydra
import wandb

from mad_td import cfgs


def fix_config(cfg, env):
    action_shape = env.get_action_space()
    observation_shape = env.get_observation_space()

    cfg.env.state_shape = observation_shape[-1]

    cfg.models.actor.output_dim = action_shape[-1]

    if cfg.algo.quantile and cfg.algo.c51:
        raise ValueError("Cannot have both quantile and c51")
    if cfg.algo.c51:
        cfg.algo.vmin = cfg.models.critic.vmin
        cfg.algo.vmax = cfg.models.critic.vmax

    try:
        bins = cfg.models.critic.num_bins
    except AttributeError:
        bins = 0
    cfg.algo.is_binned = bins > 0
    try:
        cfg.algo.num_quantile = cfg.models.critic.num_quantile
    except AttributeError:
        cfg.algo.num_quantile = bins

    # handles logging correctly
    if cfg.cluster_checkpointing:
        os.makedirs(
            f"/checkpoint/{getpass.getuser()}/{os.environ.get('SLURM_JOB_ID')}",
            exist_ok=True,
        )
        os.chdir(f"/checkpoint/{getpass.getuser()}/{os.environ.get('SLURM_JOB_ID')}")

        if cfg.get("alt_path", None) is not None:
            os.chdir(f"{cfg.alt_path}")
        cfg.alt_path = "."

    # handle inconsistent options
    if cfg.models.critic.get("add_zero_prior", False):
        assert not cfg.algo.c51, "C51 does not need a prior"
    if cfg.models.critic.get("use_symlog", False):
        assert not cfg.algo.c51, "C51 cannot use symlog"

    return cfg


@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(cfg):
    print("Guarded imports after this")

    import sys
    import traceback

    # This main is used to circumvent a bug in Hydra
    # See https://github.com/facebookresearch/hydra/issues/2664

    try:
        run(cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()


def run(cfg):
    import multiprocessing

    multiprocessing.set_start_method("fork")
    import signal

    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    import subprocess
    from omegaconf import OmegaConf
    from jax import config

    try:
        from jax.config import config
    except ImportError:
        ...  # newer jax moved jax.config.config to jax.config

    from mad_td.trainers.trainers import MultiSeedTrainer
    from mad_td.data.env import EnvConfig, make_env
    from mad_td.data.replay_buffer import ReplayBuffer
    from mad_td.utils.logging import WandBLogger

    # debug options for jitting
    config.update("jax_disable_jit", cfg.debug)

    # building config objects
    env_cfg: EnvConfig = OmegaConf.to_object(cfg.env)  # type: ignore
    env = make_env(env_cfg)

    cfg = fix_config(cfg, env)

    try:
        call = subprocess.run(
            [
                "nvidia-smi",
            ],
            capture_output=True,
        )
        print(call.stdout.decode("utf-8"))
    except BaseException as e:
        print(e)
    print(f"Loading from {cfg.alt_path}")
    print(f"Logging to {os.getcwd()}")

    print(cfg.algo)

    train_cfg: cfgs.TrainHyperparams = OmegaConf.to_object(cfg.train)  # type: ignore
    env_cfg: cfgs.EnvConfig = OmegaConf.to_object(cfg.env)  # type: ignore
    algo_cfg: cfgs.MadTdHyperparams = OmegaConf.to_object(cfg.algo)  # type: ignore

    # setup models
    critic = hydra.utils.instantiate(cfg.models.critic)
    critic_target = hydra.utils.instantiate(cfg.models.critic)
    actor = hydra.utils.instantiate(cfg.models.actor)
    encoder = hydra.utils.instantiate(cfg.models.encoder)
    encoder_target = hydra.utils.instantiate(cfg.models.encoder)
    latent_model = hydra.utils.instantiate(cfg.models.latent_model)

    replay_buffer = ReplayBuffer(
        env.get_observation_space(),
        env.get_action_space(),
        env,
        train_cfg.num_seeds,
        train_cfg.total_steps,
        max(algo_cfg.length_training_rollout, 1),
    )
    eval_replay_buffer = ReplayBuffer(
        env.get_observation_space(),
        env.get_action_space(),
        env,
        train_cfg.num_seeds,
        # factor 2 is a safety margin
        int(2 * train_cfg.total_steps * train_cfg.eval_sample_ratio),
        max(algo_cfg.length_training_rollout, 1),
    )

    logger = WandBLogger(
        cfg.logger.project,
        cfg.logger.entity,
        OmegaConf.to_container(cfg, resolve=True),
        wandb_init_path=os.path.join(cfg.alt_path, "wandb_init.txt"),
    )
    if cfg.debug:
        logger = WandBLogger(
            "debug_{}".format(cfg.logger.project),
            cfg.logger.entity,
            OmegaConf.to_container(cfg, resolve=True),
            debug=cfg.debug,
        )

    trainer = MultiSeedTrainer(
        critic,
        critic_target,
        actor,
        encoder,
        encoder_target,
        latent_model,
        replay_buffer,
        eval_replay_buffer,
        env,
        algo_cfg,
        train_cfg,
        logger,
    )

    trainer.check_pretrain("checkpoint", cfg.alt_path)
    trainer.train()
    wandb.finish(quiet=True)
    env.close()
    return 0


if __name__ == "__main__":
    main()
