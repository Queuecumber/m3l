from dataclasses import dataclass, field
from typing import Any, List, Optional
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING, DictConfig
from pathlib import Path
from hydra.utils import call
import contextlib


@dataclass
class Experiment:
    data: pl.LightningDataModule
    model: pl.LightningModule
    trainer: pl.Trainer
    name: str
    runner: Optional[DictConfig]
    mlflow: Optional[DictConfig]

    @property
    def __optional_mlflow(self):
        if self.mlflow is not None:
            import mlflow

            mlflow.set_tracking_uri(self.mlflow.tracking_uri)
            mlflow.set_experiment(self.mlflow.experiment)
            mlflow.pytorch.autolog(**self.mlflow.logging)
            return lambda: mlflow.start_run(**self.mlflow.run)

        return contextlib.nullcontext

    def fit(self) -> None:
        with self.__optional_mlflow():
            self.trainer.fit(self.model, self.data)

    def test(self) -> None:
        with self.__optional_mlflow():
            self.trainer.test(datamodule=self.data)


defaults = [{"trainer": "lightning"}]


@dataclass
class ExperimentConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    _target_: str = "mm.run.Experiment"
    data: Any = MISSING
    model: Any = MISSING
    trainer: Any = MISSING
    name: str = MISSING
    runner: Any = None
    mlflow: Any = None


cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig)

OmegaConf.register_resolver("modpath", lambda: Path(__file__).parent.absolute())