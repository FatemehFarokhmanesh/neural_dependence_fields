from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResolutionLimiter(nn.Module):

    def __init__(self, encoding: nn.Module, resolution=20):
        super().__init__()
        self.encoding = encoding
        self.resolution = resolution

    def forward(self, positions: Tensor) -> Tensor:
        positions_dummy = torch.empty_like(positions)
        positions_dummy[..., :-1] = positions[..., :-1]

        # perform linear interpolation with reduced resolution in z-direction
        # discretize z to intended resolution
        z_res = (positions[..., -1] + 1.) / 2. * (self.resolution - 1)
        # get positions for upper and lower interpolation node
        z_upper = torch.ceil(z_res)
        z_lower = torch.floor(z_res)

        # sample values at upper and lower positions
        positions_dummy[..., -1] = (2. / (self.resolution - 1.)) * z_upper - 1.
        samples_upper = self.encoding(positions_dummy)
        positions_dummy[..., -1] = (2. / (self.resolution - 1.)) * z_lower - 1.
        samples_lower = self.encoding(positions_dummy)

        # interpolate
        x = (z_res - z_lower).unsqueeze(-1)
        samples = x * samples_upper + (1. - x) * samples_lower
        return samples


def _test():
    from matplotlib import pyplot as plt

    class FeatureGrid(nn.Module):
        """
        Simple 3D feature grid, only for test purposes
        """

        def __init__(self, num_channels: int, r: Tuple[int, int, int]):
            super().__init__()
            self.register_parameter('data', nn.Parameter((torch.rand(1, num_channels, *r) - 0.5) / 10., requires_grad=True))

        def forward(self, positions: Tensor):
            grid = positions.view(1, 1, 1, *positions.shape)
            samples = F.grid_sample(self.data, grid, align_corners=True)
            samples = torch.transpose(samples[0, :, 0, 0, :], 0, 1)
            return samples

    def _to_numpy_1d(x: Tensor):
        return x.data.cpu().numpy().ravel()

    positions = torch.zeros(128, 3)
    positions[..., -1] = torch.linspace(-1, 1, 128)

    encoding = FeatureGrid(1, r=(16, 1, 1))
    samples_raw = _to_numpy_1d(encoding(positions))

    encoding_reduced = ResolutionLimiter(encoding, resolution=4)
    samples_reduced = _to_numpy_1d(encoding_reduced(positions))

    x = _to_numpy_1d(positions[:, -1])
    plt.figure()
    plt.plot(x, samples_raw, label='full resolution')
    plt.plot(x, samples_reduced,label='resolution limited')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    _test()
