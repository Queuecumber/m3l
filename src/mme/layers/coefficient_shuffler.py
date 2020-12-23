from typing import Optional, Sequence

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import fold, pad, unfold


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
