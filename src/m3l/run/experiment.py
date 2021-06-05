from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import pytorch_lightning as pl


@dataclass
class Experiment:
    data: pl.LightningDataModule
    net: pl.LightningModule
    trainer: pl.Trainer
    name: str
    cluster: Optional[Mapping[str, Any]]
    checkpoint: Optional[str]
    serializer: Optional[Callable]
    job_type: Optional[str]
    callbacks: Mapping[str, Any]

    def fit(self) -> None:
        self.trainer.fit(self.net, self.data)

        self.trainer.save_checkpoint("final.pt")

        if hasattr(self.trainer.logger.experiment, "log_artifact"):
            from wandb import Artifact

            artifact = wandb.Artifact(self.name, type="model")
            artifact.add_file("final.pt")
            self.trainer.logger.experiment.log_artifact(artifact)

    def test(self) -> None:
        self.net = type(self.net).load_from_checkpoint(self.checkpoint, map_location="cpu")
        self.trainer.test(self.net, datamodule=self.data)

    def correct(self) -> None:
        self.net = type(self.net).load_from_checkpoint(self.checkpoint, map_location="cpu")

        out_batches = self.trainer.predict(self.net, datamodule=self.data)

        for b in out_batches:
            self.serializer(b)
