import os
import sys
import torch
import itertools
import zipfile
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--pth_path', type=str, default='./experiments/models', help='path to model .pth files')
parser.add_argument('--bin_path', type=str, default='./experiments/models/bin', help='path to model .bin files')
parser.add_argument('--epoch', type=int, help='epoch size', default=300)
parser.add_argument('--n_fourier_features', type=int, default=12)
parser.add_argument('--n_channels', type=int, help='number of channels', default=12)
parser.add_argument('--spatial_dim', type=int, help='data diemnsion', default=3)
parser.add_argument('--n_encoder_layers', type=int, help='number of encoder layers', default=6)
parser.add_argument('--n_decoder_layers', type=int, help='number of decoder layers', default=6)

args_dict = vars(parser.parse_args())

epochs = args_dict['epoch']
fou = args_dict['n_fourier_features']
n_channels = args_dict['n_channels']
dim = args_dict['spatial_dim']
n_layers_encoder = args_dict['n_encoder_layers']
n_layers_decoder = args_dict['n_decoder_layers']
path = args_dict['pth_path']
bin_path = args_dict['bin_path']
if not os.path.isdir(bin_path):
    os.makedirs(bin_path)
encoder_path = f'{path}/encoder{epochs}.pth'
input_processing_path = f'{path}/input_processing{epochs}.pth'
decoder_path = f'{path}/decoder{epochs}.pth'
format = np.uint32(0)

def main():
    checkpoint_encoder = checkpoint_loader(encoder_path)
    checkpoint_grid = checkpoint_loader(input_processing_path)
    checkpoint_decoder = checkpoint_loader(decoder_path)

    cp_dict_grid = dict(checkpoint_grid)
    cp_values_grid = list(cp_dict_grid.values())
    cp_values_grid_0 = torch.flatten(cp_values_grid[0]).tolist()

    cp_dict = dict(checkpoint_encoder)
    cp_values = list(cp_dict.values())

    cp_values0_pos_nerf = cp_values[0][:, 0: dim + 2 * fou]
    cp_values0_grid = cp_values[0][:, dim + 2 * fou:]

    pad_grid = 16 - ((cp_values[0].shape[-1] + 1) % 16)
    p1d_1 = (0, 1)
    p1d_2 = (0, pad_grid)
    cp_values0_pos_nerf = torch.nn.functional.pad(cp_values0_pos_nerf, p1d_1, 'constant', 0)
    cp_values0_grid = torch.nn.functional.pad(cp_values0_grid, p1d_2, 'constant', 0)
    cp_values0 = torch.cat([cp_values0_pos_nerf, cp_values0_grid], dim=1)
    encoder_param_list = []
    cp_values_0 = torch.flatten(cp_values0).tolist()
    encoder_param_list.append(cp_values_0)

    for i in range(1, n_layers_encoder):
        cp_values_i = torch.flatten(cp_values[i]).tolist()
        encoder_param_list.append(cp_values_i)

    encoder_param_list.append(cp_values_grid_0)

    data_raw_enc = np.float32(np.array(list(itertools.chain.from_iterable(encoder_param_list))))
    #bin_writer(data_raw_enc, format, bin_path, module='encoder')

    # decoder
    cp_dict = dict(checkpoint_decoder)
    cp_values = list(cp_dict.values())
    pad_out = 16 - cp_values[5].shape[0] % 16
    
    decoder_param_list = []
    for i in range(0, n_layers_decoder-1):
        cp_values_i = torch.flatten(cp_values[i]).tolist()
        decoder_param_list.append(cp_values_i)
    
    p1d = (0, 0, 0, pad_out)
    cp_values5 = torch.nn.functional.pad(cp_values[n_layers_decoder-1], p1d, 'constant', 0)
    cp_values_5 = torch.flatten(cp_values5).tolist()
    decoder_param_list.append(cp_values_5)
    data_raw_dec = np.float32(np.array(list(itertools.chain.from_iterable(decoder_param_list))))
    #bin_writer(data_raw_dec, format, bin_path, module='decoder')

    zip_writer(data_raw_enc, data_raw_dec, path, bin_path)

def checkpoint_loader(pth):
    return torch.load(pth, map_location={'cuda:0': 'cpu'})

def bin_writer(data_raw, format, bin_path, module):
    numparam = np.uint32(data_raw.shape[0])
    draw_file = open(f'{bin_path}/data.bin', 'w+')
    data_raw.tofile(draw_file)
    header = np.array([format, numparam])
    header_file = open(f'{bin_path}/header.bin', 'w+')
    header.tofile(header_file)
    with open(f'{bin_path}/data.bin', 'rb') as f:
        data_ = f.read()
    with open(f'{bin_path}/header.bin', 'rb') as f:
        header_ = f.read()
    with open(f'{bin_path}/network_{module}.bin', 'wb') as f:
        f.write(header_ + data_)

def serialize_network_weights(weights):
    weights_bytearray = bytearray()
    weights_bytearray.extend(int(0).to_bytes(4, sys.byteorder))
    weights_bytearray.extend(int(weights.shape[0]).to_bytes(4, sys.byteorder))
    weights_bytearray.extend(weights.tobytes())
    return weights_bytearray

def read_file(path):
    with open(path, 'r') as file:
        return file.read()

def zip_writer(weights_encoder, weights_decoder, path, bin_path):
    config_encoder_data = read_file(f'{path}/config_encoder.json')
    config_decoder_data = read_file(f'{path}/config_decoder.json')
    config_data = read_file(f'{path}/config.json')

    out_path = f'{bin_path}/network.zip'
    archive = zipfile.ZipFile(out_path, 'w')
    archive.writestr('config.json', config_data)
    archive.writestr('config_encoder.json', config_encoder_data)
    archive.writestr('config_decoder.json', config_decoder_data)

    weights_encoder = serialize_network_weights(weights_encoder)
    weights_decoder = serialize_network_weights(weights_decoder)
    archive.writestr('network_encoder.bin', weights_encoder)
    archive.writestr('network_decoder.bin', weights_decoder)

if __name__ == '__main__':
    main()
