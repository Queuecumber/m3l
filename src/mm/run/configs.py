from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

experiment_defaults = [{"trainer": "lightning"}, {"serializer": None}]


@dataclass
class ExperimentConfig:
    defaults: List[Any] = field(default_factory=lambda: experiment_defaults)

    _target_: str = "mm.run.Experiment"
    data: Any = MISSING
    net: Any = MISSING
    trainer: Any = MISSING
    name: str = MISSING
    job_type: Optional[str] = None
    cluster: Any = None
    checkpoint: Optional[str] = None
    serializer: Optional[Any] = None


@dataclass
class CorrectionConfig:
    _target_: str = "mm.run.Correction"
    path: Path = MISSING
    net: Any = MISSING
    name: str = MISSING
    weights: Path = MISSING


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


@dataclass
class WandbConfig:
    _target_: str = "pytorch_lightning.loggers.WandbLogger"
    name: str = "${name}"
    save_dir: str = "."
    offline: Optional[bool] = None
    id: Optional[str] = None
    anonymous: Optional[bool] = None
    project: Optional[str] = "mm"
    log_model: Optional[bool] = None
    prefix: Optional[str] = None
    job_type: Optional[str] = "${job_type}"


cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig, provider="mm")
cs.store(group="logger", name="comet", package="trainer.logger", node=CometConfig, provider="mm")
cs.store(group="logger", name="wandb", package="trainer.logger", node=WandbConfig, provider="mm")
