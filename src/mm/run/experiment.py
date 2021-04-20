from dataclasses import dataclass

import pytorch_lightning as pl
import torch.jit


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

        if hasattr(self.trainer.logger.experiment, "log_model"):
            self.trainer.logger.experiment.log_model(self.name, f"{self.name}.pt")

    def test(self) -> None:
        pass
