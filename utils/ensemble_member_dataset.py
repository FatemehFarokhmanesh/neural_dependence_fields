import os
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from utils.sampling import CauchySampler

sampler = CauchySampler(scale=0.2, num_dims=3, min=-1, max=1.)

MEMBERS = 5
LEVELS = 20
LATITUDE = 352
LONGITUDE = 250

class EnsemblePositionDataset(Dataset):

    def __init__(self, data, num_points, grid='random'):
        self.data = data
        self.num_points = num_points
        self.grid = grid
        self.pairs = self.grid_sampler()
        self.N = self.pairs[1].shape[0]
    
    def __getitem__(self, idx):
            pos_ref, pos, in_ref, in_ = self.pairs[0][idx, :], self.pairs[1][idx, :], self.pairs[2][idx, :], self.pairs[3][idx, :]
            return pos_ref, pos, in_ref, in_
    
    def __len__(self):
        return self.N
    
    def grid_sampler(self):
        data_min = np.nanmin(self.data)
        data_max = np.nanmax(self.data)
        input = torch.from_numpy(self.data).unsqueeze(1)
        if self.grid == 'random':
            sampled_positions_ref = 2 * torch.rand(self.num_points, 3) - 1 # 3 refers to 0->x(lon), 1->y(lat), 2->z(lev)
            sampled_positions = 2 * torch.rand(self.num_points, 3) - 1
        elif self.grid == 'distance':
            sampled_positions_ref, sampled_positions = sampler.sample(self.num_points)

        positions_ref, out_ref, idx_ref = self._grid_sampler(input, sampled_positions_ref, data_min, data_max)
        positions, out, idx = self._grid_sampler(input, sampled_positions, data_min, data_max)
        idx_new = torch.logical_and(idx_ref, idx)
        positions_ref, positions = positions_ref[idx_new], positions[idx_new]
        out_ref, out = out_ref[idx_new], out[idx_new]
        return positions_ref, positions, out_ref, out

    
    def _grid_sampler(self, input, positions, min, max):
        member_grid = positions.unsqueeze(0).repeat(MEMBERS, 1, 1)  # add member dim
        grid = member_grid.unsqueeze(2).unsqueeze(2)
        output = torch.nn.functional.grid_sample(input, grid, align_corners=True)
        output_squeezed = torch.squeeze(output)
        idx = ~torch.any(output_squeezed.isnan(), dim=0)
        out_norm = (output_squeezed -min) / (max - min)
        out_norm = torch.transpose(out_norm, 0, 1)
        return positions, out_norm, idx

class SampledEnsembleDataset(Dataset):

    def __init__(self, data:list):
        self.data = data
        self.N = self.data.shape[0]
    
    def __getitem__(self, idx):
            pos_ref, pos, corr = self.data[idx, :3], self.data[idx, 3:6], self.data[idx, -1]
            return pos_ref, pos, corr
    
    def __len__(self):
        return self.N

class EnsembleMultiVarDataset(Dataset):

    def __init__(self, variables, num_points, grid='random'):
        self.variables = variables
        self.num_points = num_points
        self.grid = grid
        self.pairs = self.grid_sampler()
        self.N = self.pairs[0].shape[0]
    
    def __getitem__(self, idx):
            pos_ref, pos, in_ref, in_ = self.pairs[0][idx, :], self.pairs[1][idx, :], self.pairs[2][idx, :], self.pairs[3][idx, :]
            return pos_ref, pos, in_ref, in_
    
    def __len__(self):
        return self.N
    
    def grid_sampler(self):
        vars_min_max = [(np.nanmin(var), np.nanmax(var)) for var in self.variables]
        input_1 = torch.from_numpy(self.variables[0]).unsqueeze(1)
        input_2 = torch.from_numpy(self.variables[1]).unsqueeze(1)
        if self.grid == 'random':
            sampled_positions_ref = 2 * torch.rand(self.num_points, 3) - 1 # 3 refers to 0->x(lon), 1->y(lat), 2->z(lev)
            sampled_positions = 2 * torch.rand(self.num_points, 3) - 1
        elif self.grid == 'distance':
            sampled_positions_ref, sampled_positions = sampler.sample(self.num_points)
        positions_ref, out_ref, idx_ref = self._grid_sampler(input_1, sampled_positions_ref, vars_min_max[0][0], vars_min_max[0][1])
        positions, out, idx = self._grid_sampler(input_2, sampled_positions, vars_min_max[1][0], vars_min_max[1][1])
        idx_new = torch.logical_and(idx_ref, idx)
        positions_ref, positions = positions_ref[idx_new], positions[idx_new]
        out_ref, out = out_ref[idx_new], out[idx_new]
        return positions_ref, positions, out_ref, out

    def _grid_sampler(self, input, positions, min, max):
        member_grid = positions.unsqueeze(0).repeat(MEMBERS, 1, 1)  # add member dim
        grid = member_grid.unsqueeze(2).unsqueeze(2)
        output = torch.nn.functional.grid_sample(input, grid, align_corners=True)
        output_squeezed = torch.squeeze(output)
        idx = ~torch.any(output_squeezed.isnan(), dim=0)
        out_norm = (output_squeezed -min) / (max - min)
        out_norm = torch.transpose(out_norm, 0, 1)
        return positions, out_norm, idx



