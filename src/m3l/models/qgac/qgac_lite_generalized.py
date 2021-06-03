from typing import Optional

import pytorch_lightning as pl
import torch
from m3l.layers import RRDB  # TODO, ChannelwiseAttention, ChannelwiseThenSpatialAttention, ConvolutionalFilterManifold, JointAttention, SpatialAttention, SpatialThenChannelwiseAttention
from torch import Tensor
from torch.nn import Conv2d, ConvTranspose2d, Module, PReLU, Sequential
from torchjpeg.dct import double_nn_dct

from .weight_init import small_weight_init


class QGACLite(pl.LightningModule):
    def __init__(self, color: bool = True, attentionType: str = "none") -> None:
        super(QGACLite, self).__init__()

        self.block_y = ConvolutionalFilterManifold(in_channels=1, out_channels=32, kernel_size=8, stride=8, manifold_channels=16, post_activation=PReLU())

        self.block_enhancer_y = RRDB(kernel_size=3, channels=32, padding=1)

        self.unblock_y = ConvolutionalFilterManifold(in_channels=32, out_channels=1, kernel_size=8, stride=8, manifold_channels=16, transposed=True)

        if color:
            self.block_c = ConvolutionalFilterManifold(in_channels=2, out_channels=32, kernel_size=8, stride=8, manifold_channels=16, post_activation=PReLU())

            self.block_enhancer_lr = RRDB(kernel_size=3, channels=32, padding=1)

            self.block_resampler_420 = Sequential(ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0, bias=True), PReLU())
            self.block_resampler_422 = Sequential(ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2), padding=0, bias=True), PReLU())

            self.block_guide = ConvolutionalFilterManifold(in_channels=1, out_channels=32, kernel_size=8, stride=8, post_activation=PReLU())

            # if attentionType == "spatial":
            #     self.attention = SpatialAttention(64)
            # elif attentionType == "channelwise":
            #     self.attention = ChannelwiseAttention(64)
            # elif attentionType == "joint":
            #     self.attention = JointAttention(64)
            # elif attentionType == "spatialThenChannelwise":
            #     self.attention = SpatialThenChannelwiseAttention(64)
            # elif attentionType == "channelwiseThenSpatial":
            #     self.attention = ChannelwiseThenSpatialAttention(64)
            # else:
            #     self.attention = None

            self.block_enhancer_hr = RRDB(kernel_size=3, channels=64, padding=1)

            self.unblock_c = ConvolutionalFilterManifold(in_channels=64, out_channels=2, kernel_size=8, stride=8, manifold_channels=16, transposed=True)

        self.apply(lambda m: small_weight_init(scale=0.1, m=m))

    def forward(self, q_y: Tensor, y: Tensor, q_c: Optional[Tensor] = None, cbcr: Optional[Tensor] = None) -> Tensor:
        y = y + self.unblock_y(q_y, self.block_enhancer_y(self.block_y(q_y, y)))

        if cbcr is not None:
            c = self.block_c(q_c, cbcr)

            _, _, hc, wc = cbcr.shape
            _, _, hy, wy = y.shape

            if hc == hy and wc == wy // 2:
                c = self.block_resampler_422(self.block_enhancer_lr(c))

            elif hc == hy // 2 and wc == wy // 2:
                c = self.block_resampler_420(self.block_enhancer_lr(c))

            y_blocks = self.block_guide(q_y, y)
            c = torch.cat([y_blocks, c], dim=1)

            # if self.attention is not None:
            #     c = self.attention(c)

            c = self.unblock_c(q_c, self.block_enhancer_hr(c))

            c = c + double_nn_dct(cbcr)  # TODO support 422

            return torch.cat([y, c], dim=1)
        else:
            return y
