from dataclasses import dataclass, field
from typing import Any, List
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
from pathlib import Path

@dataclass
class Experiment:
    data: pl.LightningDataModule
    model: pl.LightningModule
    trainer: pl.Trainer
    
    def fit(self) -> None:
        self.trainer.fit(self.model, self.data)

    def test(self) -> None:
        self.trainer.test(datamodule=self.data)

defaults = [
    {"trainer": "lightning"}
]

@dataclass
class ExperimentConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    _target_: str = "mm.run.Experiment"
    data: Any = MISSING
    model: Any = MISSING
    trainer: Any = MISSING
    

cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig)

OmegaConf.register_resolver("modpath", lambda: Path(__file__).parent.absolute())