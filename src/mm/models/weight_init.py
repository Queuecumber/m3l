import torch.nn


def weight_init(scale, m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0.02, mode="fan_in")
        m.weight.data *= scale
    elif type(m) == torch.nn.PReLU:
        m.negative_slope = 0.1
        m.inplace = False
    elif type(m) == torch.nn.PReLU:
        m.weight.data = torch.Tensor([0.02])
        m.inplace = False
