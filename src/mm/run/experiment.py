from dataclasses import dataclass
from typing import Any, Optional

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

    def fit(self) -> None:
        if _module_available("comet_ml"):
            # HACK: for some reason comet complains about not being imported first, this shuts it up
            import comet_ml

            comet_ml.monkey_patching._reset_already_imported_modules()

        self.trainer.fit(self.net, self.data)

        script = self.net.to_torchscript()
        torch.jit.save(script, f"{self.name}.pt")

        if hasattr(self.trainer.logger.experiment, "log_model"):
            self.trainer.logger.experiment.log_model(self.name, f"{self.name}.pt")

    def test(self) -> None:
        if _module_available("comet_ml"):
            import comet_ml

            comet_ml.monkey_patching._reset_already_imported_modules()

        ckpt = torch.load(self.checkpoint)
        self.net.load_state_dict(ckpt["state_dict"])
        self.trainer.test(self.net, datamodule=self.data)
