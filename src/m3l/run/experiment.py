from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

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
    optimize_metrics: Sequence[str]

    def fit(self) -> Sequence[float]:
        self.trainer.fit(self.net, self.data)
        metrics = map(self.trainer.callback_metrics.get, self.optimize_metrics)
        metrics = [float(m) for m in metrics if m is not None]
        return metrics

    def test(self) -> Sequence[Mapping[str, Any]]:
        self.net = type(self.net).load_from_checkpoint(self.checkpoint, map_location="cpu")
        return self.trainer.test(self.net, datamodule=self.data)

    def correct(self) -> None:
        self.net = type(self.net).load_from_checkpoint(self.checkpoint, map_location="cpu")

        out_batches = self.trainer.predict(self.net, datamodule=self.data)

        for b in out_batches:
            self.serializer(b)
