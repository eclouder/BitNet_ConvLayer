import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

class bitConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 groups: int = 1,
                 eps: float = 1e-5,
                 bias: bool = True,
                 ):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
        self.groups = groups
        self.eps = eps

    def ste(self, x: Tensor):
        binarized_x = torch.sign(x)
        binarized_x = (binarized_x - x).detach() + x
        return binarized_x

    def quantize_activations_groupwise(self, x, b=8):

        Q_b = 2 ** (b - 1)

        group_size = x.shape[1] // self.groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[:, start_idx:end_idx]

            gamma_g = activation_group.abs().max()
            quantized_x[:, start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def binarize_weights_groupwise(self, x: Tensor):
        group_size = x.shape[1] // self.groups
        binarized_weights = torch.zeros_like(x)
        for g in range(self.groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = self.weight[:, start_idx:end_idx]
            alpha_g = weight_group.mean()
            binarized_weights[:, start_idx:end_idx] = self.ste(weight_group - alpha_g)
        return binarized_weights

    def forward(self, x: Tensor) -> Tensor:
        binarized_weights = self.binarize_weights_groupwise(self.weight)
        output = F.conv2d(x, binarized_weights)
        output = self.quantize_activations_groupwise(output)
        return output
