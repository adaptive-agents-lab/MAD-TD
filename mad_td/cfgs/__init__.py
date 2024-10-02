from hydra.core.config_store import ConfigStore

from mad_td.cfgs.data_config import (
    EnvConfig,
    GymnaxEnvConfig,
    BraxEnvConfig,
    DMCEnvConfig,
    ManiSkillEnvConfig,
    MyoSuiteEnvConfig,
)

cs = ConfigStore.instance()
cs.store(group="env", name="base_env", node=EnvConfig)
cs.store(group="env", name="gymnax_env", node=GymnaxEnvConfig)
cs.store(group="env", name="brax_env", node=BraxEnvConfig)
cs.store(group="env", name="dmc_env", node=DMCEnvConfig)
cs.store(group="env", name="maniskill_env", node=ManiSkillEnvConfig)
cs.store(group="env", name="myo_env", node=MyoSuiteEnvConfig)
