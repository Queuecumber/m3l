from typing import Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
from m3l.models.layers import RRDB, ConvolutionalFilterManifold
from torch import Tensor
from torch.nn import ConvTranspose2d, Module, PReLU, Sequential
from torch.nn.functional import l1_loss, pad
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchjpeg.dct import Stats, batch_to_images, double_nn_dct
from torchjpeg.metrics import psnr, psnrb, ssim

from ..weight_init import weight_init
from .qgac_base import QGACTrainingBatch


class PerFrequencyConvolution(Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, stride: int = 1, dilation: int = 1, transposed: bool = False, groups: int = 1, bias: int = True
    ) -> None:
        super(PerFrequencyConvolution, self).__init__()

        if transposed:
            self.filter = ConvTranspose2d(
                in_channels=in_channels * 64, out_channels=out_channels * 64, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=64, bias=bias
            )
        else:
            self.filter = Conv2d(in_channels=in_channels * 64, out_channels=out_channels * 64, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=64, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.filter(x)


class CoefficientShuffler(Module):
    def __init__(self, channels: int, direction: str = "channels") -> None:
        super(CoefficientShuffler, self).__init__()

        self._channels = channels
        self._direction = direction

    def forward(self, x: Tensor, padding: Optional[Sequence[int]] = None) -> Tensor:
        if self._direction == "channels":
            return self.channels(x)
        elif self._direction == "blocks":
            return self.blocks(x, padding)

    def channels(self, x: Tensor) -> Tensor:
        # Doing this operation efficiently requires some tricky shuffling of the data
        # We want to gather the features for each coefficient so that they are contiguous
        # across space and channels. E.g. all the 0th coefficient entries are together in
        # memory follow by all the 1st coefficients etc. This allows the entire operation
        # to be performed by a single grouped convolution.

        # First break the input into blocks
        blocks = unfold(x, kernel_size=8, stride=8)
        blocks = blocks.transpose(1, 2).contiguous().view(-1, x.shape[2] // 8, x.shape[3] // 8, self._channels, 64)

        # blocks is of shape batch size x blocks_y x blocks_x x channels x 64

        # Now move channels dimension to the front (right after batch dimension)
        blocks = blocks.transpose(2, 3).transpose(1, 2)

        # Now move the coefficient index before the channel index
        blocks = blocks.transpose(3, 4).transpose(2, 3).transpose(1, 2)

        # blocks is of shape batch size x 64 x channels x blocks_y x blocks_x
        # observe that indexing, for example, blocks[0, 0] will give all the
        # channels and all spatial locations for the 0th coefficient in the 0th
        # batch element

        # Now merge the coefficient and channel indices so that we can convolve
        blocks = blocks.contiguous().view(-1, 64 * self._channels, x.shape[2] // 8, x.shape[3] // 8)

        return blocks

    def blocks(self, x: Tensor, padding: Sequence[int]) -> Tensor:
        # This is just the inverse procedure from channels
        blocks = x.view(-1, 64, self._channels, x.shape[2], x.shape[3])
        blocks = blocks.transpose(1, 2).transpose(2, 3).transpose(3, 4)
        blocks = blocks.transpose(1, 2).transpose(2, 3)
        blocks = blocks.contiguous().view(-1, x.shape[2] * x.shape[3], self._channels * 64)
        blocks = blocks.transpose(1, 2)

        blocks = fold(blocks, kernel_size=8, stride=8, output_size=(x.shape[2] * 8, x.shape[3] * 8))

        if padding is not None:
            diffY = padding.shape[2] - blocks.shape[2]
            diffX = padding.shape[3] - blocks.shape[3]

            blocks = pad(blocks, pad=(diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        return blocks


class QGAC(pl.LightningModule):
    """
    Classic QGAC model used in "Quantization Guided JPEG Artifact Correction"

    Compared to the QGAC repository, this verison defaults to color, can only be trained end-to-end, and does not include the ablation settings.
    The input format has also been changed to match the new convention used in this repository and the code is simplified as much as possible.
    """

    def __init__(self, stats: Stats, color: bool = True) -> None:
        super(QGAC, self).__init__()
        self.stats = stats

        self.blocks_encode = BlockNet(in_channels=1, out_channels=64, n_layers=1)
        self.frequencynet = FrequencyNet(in_channels=64, out_channels=64)
        self.blocks_decode = BlockNet(in_channels=64, out_channels=64, n_layers=1)

        self.fusion = FrequencyFuser(in_channels=[64, 64, 64], fuse_channels=4)

        if color:
            self.color_net = ColorRestore(channels=64, n_layers=1)

        self.apply(lambda m: weight_init(scale=0.1, m=m))

    def __y_net(self, q_y: Tensor, y: Tensor) -> Tensor:
        blocks_e = self.blocks_encode(q_y, y)
        frequencies = self.frequencynet(blocks_e)
        blocks_d = self.blocks_decode(q_y, frequencies)
        y_r = self.fusion(blocks_d, frequencies, blocks_e)

        restored = y + y_r
        return restored

    def forward(self, q_y: Tensor, y: Tensor, q_c: Optional[Tensor], cbcr: Optional[Tensor]) -> Tensor:
        restored_y = self.__y_net(q_y, y)

        if cbcr is not None:
            r_cb = self.color_net(q_y, q_c, restored_y, cbcr[:, 0:1, ...])
            r_cr = self.color_net(q_y, q_c, restored_y, cbcr[:, 1:2, ...])

            restored_cbcr = torch.cat([r_cb, r_cr], dim=1) + double_nn_dct(cbcr)

            return torch.cat([restored_y, restored_cbcr], dim=1)
        else:
            return restored_y

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
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        return loss

    def validation_step(self, batch: QGACTrainingBatch, batch_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        y, cbcr, q_y, q_c, target, _, _ = batch

        restored = self(q_y, y, q_c, cbcr)

        target_spatial = batch_to_images(target, stats=self.stats)
        restored_spatial = batch_to_images(restored, stats=self.stats)

        psnr_e = psnr(restored_spatial, target_spatial).view(-1)
        psnrb_e = psnrb(restored_spatial, target_spatial).view(-1)
        ssim_e = ssim(restored_spatial, target_spatial).view(-1)

        self.log("val/psnr", psnr_e, prog_bar=True, sync_dist=True)
        self.log("val/psnrb", psnrb_e, sync_dist=True)
        self.log("val/ssim", ssim_e, sync_dist=True)

        return psnr_e, psnrb_e, ssim_e

    def configure_optimizers(self) -> Optimizer:
        optimizer = Adam(self.parameters(), lr=1e-4)  # TODO check if 1e-3 works here
        scheduler = CosineAnnealingLR(optimizer, 100, 1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1000,
            },
        }


class BlockNet(Module):
    def __init__(self, in_channels: int, out_channels: int, n_layers: int = 1, feature_channels: int = 256) -> None:
        super(BlockNet, self).__init__()
        self.block = ConvolutionalFilterManifold(in_channels=in_channels, out_channels=feature_channels, kernel_size=8, stride=8, post_activation=PReLU())
        self.block_enhancer = Sequential(*[RRDB(channels=feature_channels, kernel_size=3, padding=1) for _ in range(n_layers)])
        self.unblock = ConvolutionalFilterManifold(in_channels=feature_channels, out_channels=out_channels, kernel_size=8, stride=8, transposed=True, post_activation=PReLU())

    def forward(self, q: Tensor, x: Tensor) -> Tensor:
        x1 = self.block(q, x)
        x2 = self.block_enhancer(x1)
        x3 = self.unblock(q, x2)

        return x3


class FrequencyNet(Module):
    def __init__(self, in_channels: int, out_channels: int, n_layers: int = 1, feature_channels: int = 4) -> None:
        super(FrequencyNet, self).__init__()

        self.net = Sequential(
            CoefficientShuffler(channels=in_channels, direction="channels"),
            PerFrequencyConvolution(in_channels=in_channels, out_channels=feature_channels, kernel_size=3, padding=1),
            PReLU(),
            *[
                RRDB(
                    kernel_size=3,
                    channels=feature_channels,
                    conv_op=PerFrequencyConvolution,
                    padding=1,
                )
                for _ in range(n_layers)
            ],
            PerFrequencyConvolution(in_channels=feature_channels, out_channels=out_channels, kernel_size=3, padding=1),
            PReLU(),
            CoefficientShuffler(channels=out_channels, direction="blocks")
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FrequencyFuser(Module):
    def __init__(self, in_channels: int, fuse_channels: int = 4) -> None:
        super(FrequencyFuser, self).__init__()

        total_in = sum(in_channels)

        self.net = Sequential(
            CoefficientShuffler(channels=total_in, direction="channels"),
            PerFrequencyConvolution(in_channels=total_in, out_channels=fuse_channels, kernel_size=3, padding=1),
            PReLU(),
            PerFrequencyConvolution(in_channels=fuse_channels, out_channels=fuse_channels, kernel_size=3, padding=1),
            PReLU(),
            PerFrequencyConvolution(in_channels=fuse_channels, out_channels=1, kernel_size=3, padding=1),
            CoefficientShuffler(channels=1, direction="blocks"),
        )

    def forward(self, *args: Tensor) -> Tensor:
        x = torch.cat(args, 1)
        x = self.net(x)
        return x


class ColorRestore(Module):
    def __init__(self, channels: int, n_layers: int = 1):
        super(ColorRestore, self).__init__()

        self.block_c = ConvolutionalFilterManifold(in_channels=1, out_channels=32, kernel_size=8, stride=8, post_activation=PReLU())

        self.block_enhancer = Sequential(*[RRDB(kernel_size=3, channels=32, padding=1) for _ in range(n_layers)])

        self.block_doubler = Sequential(ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True), PReLU())

        self.block_y = ConvolutionalFilterManifold(in_channels=1, out_channels=32, kernel_size=8, stride=8, post_activation=PReLU())

        self.block_enhancer2 = Sequential(*[RRDB(kernel_size=3, channels=64, padding=1) for _ in range(n_layers)])

        self.unblock = ConvolutionalFilterManifold(in_channels=64, out_channels=1, kernel_size=8, stride=8, transposed=True, post_activation=PReLU())

    def forward(self, q_y: Tensor, q_c: Tensor, y: Tensor, c: Tensor) -> Tensor:
        c = self.block_c(q_c, c)
        d = self.block_enhancer(c)
        d = self.block_doubler(d)

        y = self.block_y(q_y, y)

        d = self.block_enhancer2(torch.cat([d, y], dim=1))

        d = self.unblock(q_c, d)

        return d


def safe_concat(x1: Tensor, x2: Tensor) -> Tensor:
    diffY = x2.shape[2] - x1.shape[2]
    diffX = x2.shape[3] - x1.shape[3]

    x1 = pad(x1, pad=(diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

    return torch.cat([x1, x2], 1)


def concat_perfrequency(a: Tensor, b: Tensor) -> Tensor:
    shuffler = CoefficientShuffler(channels=a.shape[1] // 64)

    a = shuffler.blocks(a, None)
    b = shuffler.blocks(b, None)

    c = safe_concat(a, b)

    shuffler._channels = c.shape[1]

    return shuffler.channels(c)
