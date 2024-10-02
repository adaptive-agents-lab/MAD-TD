from hydra.core.config_store import ConfigStore

from mad_td.cfgs.mad_td_config import MadTdHyperparams
from mad_td.cfgs.train_config import TrainHyperparams
from mad_td.cfgs.data_config import EnvConfig

cs = ConfigStore.instance()
cs.store(group="train", name="base_train", node=TrainHyperparams)
cs.store(group="env", name="base_env", node=EnvConfig)
cs.store(group="algo", name="base_mad_td", node=MadTdHyperparams)
