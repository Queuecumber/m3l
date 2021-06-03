from torch.nn import AdaptiveAvgPool2d, Conv2d, Flatten, Module, PReLU, Sequential, Softmax


class SpatialAttention(Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()

        self.net = Sequential(
            Conv2d(in_channels=channels, out_channels=16, kernel_size=1, padding=0, bias=True),
            PReLU(),
            Conv2d(in_channels=16, out_channels=channels, kernel_size=1, padding=0, bias=True),
            Flatten(2),
            Softmax(2),
        )

    def forward(self, x):
        n, c, h, w = x.shape
        w_s = self.net(x).view(n, c, h, w)
        return w_s * x


class ChannelwiseAttention(Module):
    def __init__(self, channels):
        super(ChannelwiseAttention, self).__init__()

        self.net = Sequential(
            Conv2d(in_channels=channels, out_channels=16, kernel_size=1, padding=0, bias=True),
            PReLU(),
            Conv2d(in_channels=16, out_channels=channels, kernel_size=1, padding=0, bias=True),
            AdaptiveAvgPool2d(1),
            Softmax(1),
        )

    def forward(self, x):
        w_c = self.net(x)
        return w_c * x


class JointAttention(Module):
    def __init__(self, channels) -> None:
        super(JointAttention, self).__init__()

        self.net = Sequential(
            Conv2d(in_channels=channels, out_channels=16, kernel_size=1, padding=0, bias=True),
            PReLU(),
            Conv2d(in_channels=16, out_channels=channels, kernel_size=1, padding=0, bias=True),
            Flatten(),
            Softmax(1),
        )

    def forward(self, x):
        n, c, h, w = x.shape
        w_j = self.net(x).view(n, c, h, w)

        return w_j * x


class ChannelwiseThenSpatialAttention(Module):
    def __init__(self, channels):
        super(ChannelwiseThenSpatialAttention, self).__init__()

        self.spatial = SpatialAttention(channels)
        self.channelwise = ChannelwiseAttention(channels)

    def forward(self, x):
        return self.spatial(self.channelwise(x))


class SpatialThenChannelwiseAttention(Module):
    def __init__(self, channels):
        super(SpatialThenChannelwiseAttention, self).__init__()

        self.spatial = SpatialAttention(channels)
        self.channelwise = ChannelwiseAttention(channels)

    def forward(self, x):
        return self.channelwise(self.spatial(x))
