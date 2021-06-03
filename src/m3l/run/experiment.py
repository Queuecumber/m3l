from dataclasses import dataclass
from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch.jit
from pytorch_lightning.utilities import _module_available


@dataclass
class Experiment:
    data: pl.LightningDataModule
    net: pl.LightningModule
    trainer: pl.Trainer
    name: str
    cluster: Any
    checkpoint: Optional[str]
    serializer: Optional[Callable]
    job_type: Optional[str]

    def fit(self) -> None:
        self.trainer.fit(self.net, self.data)

        trainer.save_checkpoint("final.pt")

        if hasattr(self.trainer.logger.experiment, "log_artifact"):
            from wandb import Artifact

            artifact = wandb.Artifact(self.name, type="model")
            artifact.add_file("final.pt")
            self.trainer.logger.experiment.log_artifact(artifact)

    def test(self) -> None:
        ckpt = torch.load(self.checkpoint, map_location="cpu")
        self.net.load_state_dict(ckpt["state_dict"])
        self.trainer.test(self.net, datamodule=self.data)

    def correct(self) -> None:
        ckpt = torch.load(self.checkpoint, map_location="cpu")
        self.net.load_state_dict(ckpt["state_dict"])

        out_batches = self.trainer.predict(self.net, datamodule=self.data)

        for b in out_batches:
            images, paths = b

            for i, p in zip(images, paths):
                self.serializer(i, p)
