from dataclasses import MISSING, dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@dataclass
class RunConfig:
    optimizer: DictConfig = MISSING
    lr_scheduler: DictConfig = MISSING
    dataset: DictConfig = MISSING
    model: DictConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="run", node=RunConfig)
