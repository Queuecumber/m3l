from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING, DictConfig
from pathlib import Path
import contextlib
from hydra.utils import instantiate


defaults = [{"trainer": "lightning"}]


@dataclass
class ExperimentConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    _target_: str = "mm.run.Experiment"
    data: Any = MISSING
    model: Any = MISSING
    trainer: Any = MISSING
    name: str = MISSING
    cluster: Any = None
    mlflow: Any = None


@dataclass
class Experiment:
    data: pl.LightningDataModule
    model: pl.LightningModule
    trainer: pl.Trainer
    name: str
    cluster: Optional[DictConfig]
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

    @staticmethod
    def run_experiment(cfg: ExperimentConfig, target: Callable[[Any], Any]):
        def target_wrap(cfg: ExperimentConfig, job_env: Any):
            experiment: Experiment = instantiate(cfg)

            return target(experiment)

        if cfg.cluster is not None:
            from .slurm import slurm_launch

            slurm_launch(cfg, target_wrap)
        else:
            target_wrap(cfg, None)


cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig)

OmegaConf.register_resolver("modpath", lambda: Path(__file__).parent.absolute())