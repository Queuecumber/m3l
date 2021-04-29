from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

experiment_defaults = [{"trainer": "lightning"}]


@dataclass
class ExperimentConfig:
    defaults: List[Any] = field(default_factory=lambda: experiment_defaults)

    _target_: str = "mm.run.Experiment"
    data: Any = MISSING
    net: Any = MISSING
    trainer: Any = MISSING
    name: str = MISSING


@dataclass
class CometConfig:
    _target_: str = "pytorch_lightning.loggers.CometLogger"
    api_key: str = MISSING
    save_dir: str = "."
    project_name: Optional[str] = "mm"
    workspace: Optional[str] = None
    rest_api_key: Optional[str] = None
    experiment_name: str = "${name}"
    experiment_key: Optional[str] = None
    offline: bool = False
    prefix: str = ""


cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig, provider="mm")
cs.store(group="logger", name="comet", package="trainer.logger", node=CometConfig, provider="mm")
