import math
from typing import Union, Tuple
from matplotlib import pyplot as plt

import numpy as np
import torch
from torch import Tensor, nn

from latent_features.marginal.multi_res.hash_features import HashedFeatureGrid
from latent_features.marginal.multi_res.interface import IFeatureModule
from latent_features.marginal.multi_res.features import FeatureGrid
from latent_features.marginal.multi_res.torch_init import DefaultInitializer


class MultiResolutionFeatures(IFeatureModule):
    """
    Implementation of hash-based multi-resolution features
    Reference: https://arxiv.org/abs/2201.05989
    """

    def evaluate(self,  positions: Tensor, time: Tensor, member: Tensor) -> Tensor:
        features = [f.evaluate(positions, time, member) for f in self.features]
        return torch.cat(features, dim=-1)

    def uses_positions(self) -> bool:
        return True

    def uses_member(self) -> bool:
        return False

    def uses_time(self) -> bool:
        return False

    def __init__(self, *features: Union[HashedFeatureGrid, FeatureGrid], debug=False):
        dimension = features[0].dimension
        assert np.all([f.dimension == dimension for f in features])
        num_channels = int(np.sum([f.num_channels() for f in features]))
        super(MultiResolutionFeatures, self).__init__(dimension, num_channels, debug)
        self.features = nn.ModuleList([f.set_debug(False) for f in features])

    @classmethod
    def from_initializer(
            cls,
            initializer,
            coarse_resolution: Tuple[int, ...], fine_resolution: Tuple[int, ...],
            num_levels: int, num_nodes: int, num_channels: int,
            grid_width=None, grid_offset=None, hash_primes=None,
            debug=False, device=None, dtype=None
    ):
        grid_channels = int(num_channels // num_levels)
        grids = []

        for l in range(num_levels):
            def compute_resolution(c, f):
                b = math.exp((math.log(f) - math.log(c)) / (num_levels - 1))
                return int(math.floor(c * b ** l))
            resolution = tuple([compute_resolution(c, f) for c, f in zip(coarse_resolution, fine_resolution)])
            if np.prod(tuple(x+1 for x in resolution)) <= num_nodes:
                current_grid = FeatureGrid.from_initializer(
                    initializer, resolution, grid_channels,
                    debug=False, device=device, dtype=dtype
                )
            else:
                current_grid = HashedFeatureGrid.from_initializer(
                    initializer, resolution, num_nodes, grid_channels,
                    grid_width=grid_width, grid_offset=grid_offset, hash_primes=hash_primes,
                    debug=False, device=device, dtype=dtype
                )
            grids.append(current_grid)
        return cls(*grids, debug=debug)


def _test():
    feature_grid = MultiResolutionFeatures.from_initializer(
        DefaultInitializer(), (4, 4, 4), (512, 512, 512), 8, 2**9, 8
    )
    positions = torch.ones(250, 3) * 0.5
    positions[:, 0] = torch.linspace(0, 1, 250)
    features = feature_grid.evaluate(positions, None, None)
    features = features[:, 0].data.cpu().numpy()
    plt.scatter(np.arange(250), features)
    plt.show()
    print('Finished!!!')


if __name__ == '__main__':
    _test()
