import math
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor


class _DistanceSampler(object):

    def __init__(self, num_dims=3, min=0., max=1., device=None):
        self.num_dims = int(num_dims)
        self.min = float(min)
        self.max = float(max)
        self.device = device

    def sample_distances(self, num_samples: int) -> Tensor:
        raise NotImplementedError()

    def _sample_random_positions(self, num_samples: int) -> Tensor:
        return torch.rand(num_samples, self.num_dims, device=self.device)

    def _transform_bounds(self, x: Tensor) -> Tensor:
        return self.min + (self.max - self.min) * x

    def sample(self, num_samples) -> Tuple[Tensor, Tensor]:
        reference = self._sample_random_positions(num_samples)
        distances = self.sample_distances(num_samples)
        upper = torch.clip(reference + distances, max=1.)
        lower = torch.clip(reference - distances, min=0.)
        extent = upper - lower
        query = lower + extent * self._sample_random_positions(num_samples)
        return self._transform_bounds(reference), self._transform_bounds(query)


class ExponentialSampler(_DistanceSampler):

    def __init__(self, scale=1., num_dims=3, min=0., max=1., device=None):
        super().__init__(num_dims, min, max, device)
        self.gamma = float(scale)

    def sample_distances(self, num_samples: int) -> Tensor:
        return - torch.log1p(- torch.rand(num_samples, 1, device=self.device)) * self.gamma


class GaussianSampler(_DistanceSampler):

    _sqrt2 = math.sqrt(2.)

    def __init__(self, scale=1., num_dims=3, min=0., max=1., device=None):
        super().__init__(num_dims, min, max, device)
        self.sigma = float(scale)

    def sample_distances(self, num_samples: int) -> Tensor:
        return self.sigma * self._sqrt2 * torch.erfinv(torch.rand(num_samples, 1, device=self.device))


class CauchySampler(_DistanceSampler):

    def __init__(self, scale=1., num_dims=3, min=0., max=1., device=None):
        super().__init__(num_dims, min, max, device)
        self.gamma = float(scale)

    def sample_distances(self, num_samples: int) -> Tensor:
        return self.gamma * torch.tan(np.pi * torch.rand(num_samples, 1, device=self.device) / 2.)


class UniformSampler(object):

    def __init__(self, num_dims: int = 3, min=-1., max=1., device=None):
        self.num_dims = int(num_dims)
        self.device = device
        self.max = float(max)
        self.min = float(min)

    def _rescale(self, x: Tensor) -> Tensor:
        return (self.max - self.min) * x + self.min

    def _sample_random_positions(self, num_samples: int, rescale=False) -> Tensor:
        samples = torch.rand(num_samples, self.num_dims, device=self.device)
        if rescale:
            return self._rescale(samples)
        return samples

    def sample(self, num_samples: int) -> Tuple[Tensor, Tensor]:
        return self._sample_random_positions(num_samples, rescale=True), self._sample_random_positions(num_samples, rescale=True)

