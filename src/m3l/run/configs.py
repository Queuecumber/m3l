import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

experiment_defaults = [{"trainer": "lightning"}]
callback_defaults = {"progress": {"_target_": "m3l.run.M3LProgress"}, "lr_monitor": {"_target_": "pytorch_lightning.callbacks.LearningRateMonitor"}}


@dataclass
class ExperimentConfig:
    defaults: List[Any] = field(default_factory=lambda: experiment_defaults)

    _target_: str = "m3l.run.Experiment"
    data: Any = MISSING
    net: Any = MISSING
    trainer: Any = MISSING
    name: str = MISSING
    job_type: Optional[str] = None
    cluster: Optional[Dict[str, Any]] = None
    checkpoint: Optional[str] = None
    serializer: Optional[Any] = None
    callbacks: Dict[str, Any] = field(default_factory=lambda: callback_defaults)
    optimize_metrics: List[str] = field(default_factory=list)


@dataclass
class WandbConfig:
    _target_: str = "pytorch_lightning.loggers.WandbLogger"
    name: str = "${name}"
    save_dir: str = "."
    offline: Optional[bool] = None
    id: Optional[str] = None
    anonymous: Optional[bool] = None
    project: Optional[str] = "m3l"
    log_model: Optional[bool] = None
    prefix: Optional[str] = None
    job_type: Optional[str] = "${job_type}"


cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig, provider="m3l")
cs.store(group="logger", name="wandb", package="trainer.logger", node=WandbConfig, provider="m3l")


def customize_args() -> None:
    if "--cluster-launch" in sys.argv:
        sys.argv[sys.argv.index("--cluster-launch")] = "--multirun"
