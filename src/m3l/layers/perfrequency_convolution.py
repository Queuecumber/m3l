from torch import Tensor
from torch.nn import Conv2d, ConvTranspose2d, Module


class PerFrequencyConvolution(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, stride: int = 1, dilation: int = 1, transposed: bool = False, groups: int = 1, bias: int = True) -> None:
        super(PerFrequencyConvolution, self).__init__()

        if transposed:
            self.filter = ConvTranspose2d(
                in_channels=in_channels * 64, out_channels=out_channels * 64, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=64, bias=bias
            )
        else:
            self.filter = Conv2d(
                in_channels=in_channels * 64, out_channels=out_channels * 64, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=64, bias=bias
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.filter(x)
