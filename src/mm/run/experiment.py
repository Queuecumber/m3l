from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import pytorch_lightning as pl
import torch.jit
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import _module_available

defaults = [{"trainer": "lightning"}]


@dataclass
class ExperimentConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    _target_: str = "mm.run.Experiment"
    data: Any = MISSING
    net: Any = MISSING
    trainer: Any = MISSING
    name: str = MISSING


@dataclass
class Experiment:
    data: pl.LightningDataModule
    net: pl.LightningModule
    trainer: pl.Trainer
    name: str

    def fit(self) -> None:
        self.trainer.fit(self.net, self.data)

        script = self.net.to_torchscript()
        torch.jit.save(script, f"{self.name}.pt")

        if _module_available("wandb") and isinstance(self.trainer.logger, WandbLogger):
            import wandb

            trained_model_artifact = wandb.Artifact("trained_model", type="model", description=f"Trained {self.name} model")
            trained_model_artifact.add_file(f"{self.name}.pt")
            self.trainer.logger.experiment.log_artifact(trained_model_artifact)

    def test(self) -> None:
        self.trainer.test(datamodule=self.data)


cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig)

OmegaConf.register_resolver("modpath", lambda: Path(__file__).parent.absolute())
