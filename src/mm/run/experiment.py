from dataclasses import dataclass, field
from typing import Any, List, Optional
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING, DictConfig
from pathlib import Path

defaults = [{"trainer": "lightning"}]


@dataclass
class ExperimentConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    _target_: str = "mm.run.Experiment"
    data: Any = MISSING
    model: Any = MISSING
    trainer: Any = MISSING
    name: str = MISSING


@dataclass
class Experiment:
    data: pl.LightningDataModule
    model: pl.LightningModule
    trainer: pl.Trainer
    name: str

    def fit(self) -> None:
        self.trainer.fit(self.model, self.data)

    def test(self) -> None:
        self.trainer.test(datamodule=self.data)


cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig)

OmegaConf.register_resolver("modpath", lambda: Path(__file__).parent.absolute())