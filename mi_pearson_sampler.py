import os
import json
import glob
import argparse
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.ensemble_member_dataset import EnsemblePositionDataset, EnsembleMultiVarDataset, MEMBERS
from utils.pearson_correlation import pearson_correlation_batch_cuda
import pycoriander

parser = argparse.ArgumentParser()
parser.add_argument('--source_path', type=str, default='/mnt/data2/ensemble/1000_member', help='path to ensemble dataset nc files')
parser.add_argument('--indices', type=str, default=None, help='indices to substitute in \'--source_path\', e.g., for path \'file_%04i.nc\' it could be \'range(1,101)\'')
parser.add_argument('--num_train', type=int, default=1000000)
parser.add_argument('--num_validation', type=int, help='batch size', default=200000)
parser.add_argument('--num_samples', type=int, help='resampling iteration', default=300)
parser.add_argument('--varname', type=str, default='u', help='name of the variable to load from the .nc files')

args_dict = vars(parser.parse_args())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
multi_var = False
# SOURCE_PATH = '/mnt/data2/ensemble/1000_member'
# NUM_POINTS_TRAIN = 1000000
# NUM_POINTS_VAL = 200000
# n_samples = 300

SOURCE_PATH = args_dict['source_path']
INDICES = args_dict['indices']
VARNAME = args_dict['varname']
NUM_POINTS_TRAIN = args_dict['num_train']
NUM_POINTS_VAL = args_dict['num_validation']
n_samples = args_dict['num_samples']
EXPERIMENT_PATH ='./sampled_data'

if not os.path.isdir(EXPERIMENT_PATH):
    os.makedirs(os.path.abspath(EXPERIMENT_PATH))

TRAIN_DIR = f'{EXPERIMENT_PATH}/train'
VALIDATION_DIR = f'{EXPERIMENT_PATH}/validation'

if not os.path.isdir(TRAIN_DIR):
    os.makedirs(os.path.abspath(TRAIN_DIR))

if not os.path.isdir(VALIDATION_DIR):
    os.makedirs(os.path.abspath(VALIDATION_DIR))

if INDICES is not None:
    start = 0
    stop = None
    step = 1
    range_str = INDICES
    range_str = range_str[range_str.find('(') + 1:range_str.find(')')]
    range_str = range_str.split(',')
    range_str = [s.strip() for s in range_str]
    if len(range_str) == 1:
        stop = int(range_str[0])
    elif len(range_str) == 2:
        start = int(range_str[0])
        stop = int(range_str[1])
    elif len(range_str) == 3:
        start = int(range_str[0])
        stop = int(range_str[1])
        step = int(range_str[2])
    else:
        raise Exception('invalid indices string')
    print(f'using range({start}, {stop}, {step})')
    members_path = []
    for i in range(start, stop, step):
        members_path.append(SOURCE_PATH % i)
else:
    members_path = sorted(
        glob.glob(os.path.join(SOURCE_PATH, '*.nc')), reverse=False)
members_path = members_path[:MEMBERS]
ensemble_data = xr.open_mfdataset(members_path,
                                  concat_dim='members', combine='nested')
if 'time' in ensemble_data[VARNAME].dims:
    var_1 = ensemble_data[VARNAME].isel(time=5).values
else:
    var_1 = ensemble_data[VARNAME].values
if multi_var:
    var_2 = ensemble_data[VARNAME].isel.values
    variables = [var_1, var_2]
# multivar
if multi_var:
    dataset_validation = EnsembleMultiVarDataset(variables, NUM_POINTS_VAL)
    validation_loader = DataLoader(dataset_validation, batch_size=NUM_POINTS_VAL, num_workers=4)
    for i, batch in enumerate(validation_loader):
        positions_ref, positions, inputs_ref, inputs = batch
        positions_ref = positions_ref.to(device)
        positions = positions.to(device)
        inputs_ref = inputs_ref.to(device)
        inputs = inputs.to(device)
        corr_true = pearson_correlation_batch_cuda(inputs_ref, inputs)
        corr_true = corr_true.unsqueeze(1)
        mi_true = pycoriander.mutual_information_kraskov(inputs_ref, inputs, 10)
        mi_true = mi_true.unsqueeze(1)
        positions_ref_xr = xr.DataArray(positions_ref.cpu().detach().numpy(), name='pos_ref', dims=('samples', 'coords'))
        positions_ref_xr.to_netcdf(f'{VALIDATION_DIR}/positions_ref_val.nc')
        positions_xr = xr.DataArray(positions.cpu().detach().numpy(), name='pos', dims=('samples', 'coords'))
        positions_xr.to_netcdf(f'{VALIDATION_DIR}/positions_val.nc')
        mi_xr = xr.DataArray(mi_true.cpu().detach().numpy(), name='mi', dims=('samples', 'mi'))
        mi_xr.to_netcdf(f'{VALIDATION_DIR}/mi_val.nc')
        corr_xr = xr.DataArray(corr_true.cpu().detach().numpy(), name='pearson', dims=('samples', 'corr'))
        corr_xr.to_netcdf(f'{VALIDATION_DIR}/pearson_val.nc')
    for n in range(n_samples):
        dataset_train = EnsembleMultiVarDataset(variables, NUM_POINTS_TRAIN)
        train_loader = DataLoader(dataset_train, batch_size=NUM_POINTS_TRAIN, num_workers=4)
        for i, batch in enumerate(train_loader):
            positions_ref, positions, inputs_ref, inputs = batch
            positions_ref = positions_ref.to(device)
            positions = positions.to(device)
            inputs_ref = inputs_ref.to(device)
            inputs = inputs.to(device)
            corr_true = pearson_correlation_batch_cuda(inputs_ref, inputs)
            corr_true = corr_true.unsqueeze(1)
            mi_true = pycoriander.mutual_information_kraskov(inputs_ref, inputs, 10)
            mi_true = mi_true.unsqueeze(1)
            positions_ref_xr = xr.DataArray(positions_ref.cpu().detach().numpy(), name='pos_ref', dims=('samples', 'coords'))
            positions_ref_xr.to_netcdf(f'{TRAIN_DIR}/positions_ref_train{n}.nc')
            positions_xr = xr.DataArray(positions.cpu().detach().numpy(), name='pos', dims=('samples', 'coords'))
            positions_xr.to_netcdf(f'{TRAIN_DIR}/positions_train{n}.nc')
            mi_xr = xr.DataArray(mi_true.cpu().detach().numpy(), name='mi', dims=('samples', 'mi'))
            mi_xr.to_netcdf(f'{TRAIN_DIR}/mi_train{n}.nc')
            corr_xr = xr.DataArray(corr_true.cpu().detach().numpy(), name='pearson', dims=('samples', 'corr'))
            corr_xr.to_netcdf(f'{TRAIN_DIR}/pearson_train{n}.nc')
# singlevar
else:
    dataset_validation = EnsemblePositionDataset(var_1, NUM_POINTS_VAL, grid='random')
    validation_loader = DataLoader(dataset_validation, batch_size=NUM_POINTS_VAL, num_workers=4)
    for i, batch in enumerate(validation_loader):
        positions_ref, positions, inputs_ref, inputs = batch
        positions_ref = positions_ref.to(device)
        positions = positions.to(device)
        inputs_ref = inputs_ref.to(device)
        inputs = inputs.to(device)
        corr_true = pearson_correlation_batch_cuda(inputs_ref, inputs)
        corr_true = corr_true.unsqueeze(1)
        mi_true = pycoriander.mutual_information_kraskov(inputs_ref, inputs, 10)
        mi_true = mi_true.unsqueeze(1)
        positions_ref_xr = xr.DataArray(positions_ref.cpu().detach().numpy(), name='pos_ref', dims=('samples', 'coords'))
        positions_ref_xr.to_netcdf(f'{VALIDATION_DIR}/positions_ref_val.nc')
        positions_xr = xr.DataArray(positions.cpu().detach().numpy(), name='pos', dims=('samples', 'coords'))
        positions_xr.to_netcdf(f'{VALIDATION_DIR}/positions_val.nc')
        mi_xr = xr.DataArray(mi_true.cpu().detach().numpy(), name='mi', dims=('samples', 'mi'))
        mi_xr.to_netcdf(f'{VALIDATION_DIR}/mi_val.nc')
        corr_xr = xr.DataArray(corr_true.cpu().detach().numpy(), name='pearson', dims=('samples', 'corr'))
        corr_xr.to_netcdf(f'{VALIDATION_DIR}/pearson_val.nc')
    for n in range(n_samples):
        pos_ref_list = []
        pos_list = []
        mi_list = []
        corr_list = []
        dataset_train = EnsemblePositionDataset(var_1, NUM_POINTS_TRAIN, grid='random')
        train_loader = DataLoader(dataset_train, batch_size=int(NUM_POINTS_TRAIN / 2), num_workers=4)
        for i, batch in enumerate(train_loader):
            positions_ref, positions, inputs_ref, inputs = batch
            positions_ref = positions_ref.to(device)
            positions = positions.to(device)
            inputs_ref = inputs_ref.to(device)
            inputs = inputs.to(device)
            corr_true = pearson_correlation_batch_cuda(inputs_ref, inputs)
            corr_true = corr_true.unsqueeze(1)
            mi_true = pycoriander.mutual_information_kraskov(inputs_ref, inputs, 10)
            mi_true = mi_true.unsqueeze(1)

            pos_ref_list.append(positions_ref)
            pos_list.append(positions)
            corr_list.append(corr_true)
            mi_list.append(mi_true)

        positions_ref = torch.cat(pos_ref_list, dim=0)
        positions = torch.cat(pos_list, dim=0)
        corr_true = torch.cat(corr_list, dim=0)
        mi_true = torch.cat(mi_list, dim=0)

        positions_ref_xr = xr.DataArray(positions_ref.cpu().detach().numpy(), name='pos_ref', dims=('samples', 'coords'))
        positions_ref_xr.to_netcdf(f'{TRAIN_DIR}/positions_ref_train{n}.nc')
        positions_xr = xr.DataArray(positions.cpu().detach().numpy(), name='pos', dims=('samples', 'coords'))
        positions_xr.to_netcdf(f'{TRAIN_DIR}/positions_train{n}.nc')
        mi_xr = xr.DataArray(mi_true.cpu().detach().numpy(), name='mi', dims=('samples', 'mi'))
        mi_xr.to_netcdf(f'{TRAIN_DIR}/mi_train{n}.nc')
        corr_xr = xr.DataArray(corr_true.cpu().detach().numpy(), name='pearson', dims=('samples', 'corr'))
        corr_xr.to_netcdf(f'{TRAIN_DIR}/pearson_train{n}.nc')