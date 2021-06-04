from collections import defaultdict
from typing import Callable, Iterator, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.distributed
from m3l.models.layers import RRDB, ConvolutionalFilterManifold
from torch import Tensor
from torch.nn import ConvTranspose2d, Parameter, PReLU, Sequential
from torch.nn.functional import l1_loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchjpeg.data import crop_batch
from torchjpeg.dct import Stats, batch_to_images, double_nn_dct
from torchjpeg.metrics import psnr, psnrb, ssim

from ..loghelper import LogHelper
from ..weight_init import small_weight_init
from .qgac_base import QGACTrainingBatch


class QGACCrab(LogHelper, pl.LightningModule):
    """
    QGAC model from "Analysing and Mitigating Compression Defects in Deep Learning"
    """

    def __init__(self, stats: Stats, optimizer: Callable[[Iterator[Parameter]], Optimizer], scheduler: Callable[[Optimizer], _LRScheduler]) -> None:
        super(QGACCrab, self).__init__()
        self.stats = stats
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.save_hyperparameters()

        self.block_y = ConvolutionalFilterManifold(
            in_channels=1,
            out_channels=256,
            kernel_size=8,
            stride=8,
            manifold_channels=16,
            post_activation=PReLU(),
        )
        self.block_enhancer_y = torch.nn.Sequential(*[RRDB(channels=256, kernel_size=3, padding=1) for _ in range(3)])
        self.unblock_y = ConvolutionalFilterManifold(
            in_channels=256,
            out_channels=1,
            kernel_size=8,
            stride=8,
            manifold_channels=16,
            transposed=True,
        )

        self.block_c = ConvolutionalFilterManifold(
            in_channels=2,
            out_channels=32,
            kernel_size=8,
            stride=8,
            manifold_channels=16,
            post_activation=PReLU(),
        )
        self.block_enhancer_lr = RRDB(kernel_size=3, channels=32, padding=1)
        self.block_doubler = Sequential(
            ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            PReLU(),
        )
        self.block_guide = ConvolutionalFilterManifold(
            in_channels=1,
            out_channels=32,
            kernel_size=8,
            stride=8,
            post_activation=PReLU(),
        )
        self.block_enhancer_hr = RRDB(kernel_size=3, channels=64, padding=1)
        self.unblock_c = ConvolutionalFilterManifold(
            in_channels=64,
            out_channels=2,
            kernel_size=8,
            stride=8,
            manifold_channels=16,
            transposed=True,
        )

        self.apply(lambda m: small_weight_init(scale=0.1, m=m))

    def forward(
        self,
        q_y: Tensor,
        y: Tensor,
        q_c: Optional[Tensor] = None,
        cbcr: Optional[Tensor] = None,
    ) -> Tensor:
        y = y + self.unblock_y(q_y, self.block_enhancer_y(self.block_y(q_y, y)))

        if q_c is not None and cbcr is not None and cbcr.numel() > 0:
            c = self.block_doubler(self.block_enhancer_lr(self.block_c(q_c, cbcr)))

            y_blocks = self.block_guide(q_y, y)
            c = torch.cat([y_blocks, c], dim=1)

            c = self.unblock_c(q_c, self.block_enhancer_hr(c))

            c = c + double_nn_dct(cbcr)

            return torch.cat([y, c], dim=1)
        else:
            return y

    def training_step(self, batch: QGACTrainingBatch, batch_idx: int) -> Tensor:
        y, cbcr, q_y, q_c, target, _, _ = batch

        restored = self(q_y, y, q_c, cbcr)

        target_spatial = batch_to_images(target, stats=self.stats)
        restored_spatial = batch_to_images(restored, stats=self.stats)

        l1_e = l1_loss(restored_spatial, target_spatial)
        ssim_e = ssim(restored_spatial, target_spatial).mean()

        loss = l1_e - 0.05 * ssim_e  # Minimize l1 error and maximize SSIM

        self.log("train/loss", loss, sync_dist=True)
        self.log("train/ssim", ssim_e, sync_dist=True)
        self.log("train/l1", l1_e, sync_dist=True)

        return loss

    def training_epoch_end(self, training_step_outputs):
        self.log("train/lr", self.optimizers().optimizer.param_groups[0]["lr"], prog_bar=True)

    def validation_step(self, batch: QGACTrainingBatch, batch_idx: int):
        y, cbcr, q_y, q_c, target, sizes, _ = batch

        restored = self(q_y, y, q_c, cbcr)

        target_spatial = batch_to_images(target, stats=self.stats)
        restored_spatial = batch_to_images(restored, stats=self.stats)

        target_seq = crop_batch(target_spatial, sizes)
        restored_seq = crop_batch(restored_spatial, sizes)

        psnr_e = torch.cat([psnr(r.unsqueeze(0), t.unsqueeze(0)).view(-1) for r, t in zip(restored_seq, target_seq)])
        psnrb_e = torch.cat([psnrb(r.unsqueeze(0), t.unsqueeze(0)).view(-1) for r, t in zip(restored_seq, target_seq)])
        ssim_e = torch.cat([ssim(r.unsqueeze(0), t.unsqueeze(0)).view(-1) for r, t in zip(restored_seq, target_seq)])

        self.log("val/psnr", psnr_e, prog_bar=True, sync_dist=True)
        self.log("val/psnrb", psnrb_e, sync_dist=True)
        self.log("val/ssim", ssim_e, sync_dist=True)

        if batch_idx == 0:
            self.log_image("val/restored", restored_seq[0])

    def test_step(self, batch: QGACTrainingBatch, batch_idx: int, dataloader_idx: int):
        y, cbcr, q_y, q_c, target, sizes, _ = batch

        restored = self(q_y, y, q_c, cbcr)

        target_spatial = batch_to_images(target, stats=self.stats)
        restored_spatial = batch_to_images(restored, stats=self.stats)

        target_seq = crop_batch(target_spatial, sizes)
        restored_seq = crop_batch(restored_spatial, sizes)

        psnr_e = torch.cat([psnr(r.unsqueeze(0), t.unsqueeze(0)).view(-1) for r, t in zip(restored_seq, target_seq)])
        psnrb_e = torch.cat([psnrb(r.unsqueeze(0), t.unsqueeze(0)).view(-1) for r, t in zip(restored_seq, target_seq)])
        ssim_e = torch.cat([ssim(r.unsqueeze(0), t.unsqueeze(0)).view(-1) for r, t in zip(restored_seq, target_seq)])

        if batch_idx == 0:
            name, quality = self.trainer.datamodule.test_set_idx[dataloader_idx]
            self.log_image("test/examples", restored_seq[0], caption=f"{name}, quality: {quality}")

        output = {"psnr": psnr_e, "psnrb": psnrb_e, "ssim": ssim_e}

        return output

    def test_epoch_end(self, outputs) -> None:
        from pandas import DataFrame

        columns = ["Dataset", "Quality", "PSNR", "PSNR-B", "SSIM"]

        tables = defaultdict(lambda: DataFrame(columns=columns))

        for dso, (name, quality) in zip(outputs, self.trainer.datamodule.test_set_idx):
            row = [name, quality] + [torch.cat([d[key] for d in dso]).mean().item() for key in dso[0].keys()]
            tables[name] = tables[name].append(DataFrame([row], columns=columns))

        print(tables)

        for name, table in tables.items():
            self.log_table(f"test/{name}", table)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: Optional[int]) -> Tuple[Sequence[Tensor], Sequence[str]]:
        y, cbcr, q_y, q_c, path, sizes = batch

        restored = self(q_y, y, q_c, cbcr)

        restored_spatial = batch_to_images(restored, stats=self.stats)
        restored_seq = crop_batch(restored_spatial, sizes)

        for r, p in zip(restored_seq, path):
            self.log_image("correct/restored", r, caption=p)

        return restored_seq, path

    def configure_optimizers(self) -> Tuple[Sequence[Optimizer], Sequence[_LRScheduler]]:
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(optimizer=optimizer)
        return [optimizer], [scheduler]
