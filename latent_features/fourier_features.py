from typing import Optional

import numpy as np
from numpy import pi as PI
import torch
from torch import nn, Tensor


class IComponentProcessor(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(IComponentProcessor, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

    def in_channels(self):
        return self._in_channels

    def out_channels(self):
        return self._out_channels


class IFourierFeatures(IComponentProcessor):

    def __init__(self, fourier_matrix: Tensor):
        in_channels, half_out_channels = fourier_matrix.shape
        super(IFourierFeatures, self).__init__(in_channels, 2 * half_out_channels)
        self.register_buffer('fourier_matrix', fourier_matrix)
        self.idx_sin = torch.arange(0, half_out_channels * 2, 2).tolist()
        self.idx_cos = torch.arange(1, half_out_channels * 2, 2).tolist()

    def forward(self, x: Tensor):
        assert x.shape[-1] == self.in_channels()
        f2 = torch.matmul(x, self.fourier_matrix)
        f2_cos = torch.cos(f2.cpu())
        f2_sin = torch.sin(f2.cpu())
        out = torch.zeros(f2.shape[0], f2.shape[1] * 2)
        out[:, self.idx_sin] = f2_sin
        out[:, self.idx_cos] = f2_cos
        # m = torch.cat([torch.cos(f2), torch.sin(f2)], dim=-1)
        return out

    def get_fourier_matrix(self):
        m = self.fourier_matrix.t()
        return self.fourier_matrix.t()


class RandomFourierFeatures(IFourierFeatures):

    def __init__(self, in_channels: int, num_fourier_features: int, std: Optional[float] = 1.):
        B = 2. * PI * torch.normal(0, std, (num_fourier_features, in_channels))
        super(RandomFourierFeatures, self).__init__(B.t())


class NerfFourierFeatures(IFourierFeatures):

    def __init__(self, in_channels: int, num_fourier_features: int):
        num_blocks = int(np.ceil(num_fourier_features / in_channels))
        B = [2 ** i * torch.eye(in_channels, in_channels) for i in range(num_blocks)]
        B = 1. * PI * torch.cat(B, dim=0)[:num_fourier_features, :]
        B_x = torch.cat([B[i, :] for i in range(0, num_fourier_features, in_channels)], dim=0)
        B_x = torch.reshape(B_x, (num_blocks, in_channels))
        B_y = torch.cat([B[i, :] for i in range(1, num_fourier_features, in_channels)], dim=0)
        B_y = torch.reshape(B_y, (num_blocks, in_channels))
        B_z = torch.cat([B[i, :] for i in range(2, num_fourier_features, in_channels)], dim=0)
        B_z = torch.reshape(B_z, (num_blocks, in_channels))
        B = torch.cat([B_x, B_y, B_z], dim=0)
        super(NerfFourierFeatures, self).__init__(B.t())

