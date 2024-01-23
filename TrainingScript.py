import os
import json
import shutil
import argparse
import xarray as xr
import torch
import torch.nn as nn
import tinycudann as tcnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from latent_features.fourier_features import NerfFourierFeatures
from networks.srn_mine_net import PytorchDecoder, PytorchEncoder, MainDecoderPosition, SymmNetAdd, EncoderDecoder, SymmNetAddDiff, Multiplier
from networks.input_processing import InputProcessingPosition
from utils.batch_loader import BatchLoader
from utils.ensemble_member_dataset import SampledEnsembleDataset
from utils.ProgressBar import ProgressBar
from utils.WelfordStatisticsTracker import WelfordStatisticsTracker
from utils.pearson_correlation import pearson_correlation_batch_cuda
from configs import ProjectConfigs

parser = argparse.ArgumentParser()
parser.add_argument('--symm', type=str, default='mul', help='method to demonstrate the use of merger')
parser.add_argument('--lr', type=float, help='learning rate', default=0.0008) #choices=[0.01, 0.001, 0.0001]
parser.add_argument('--batch', type=int, help='batch size', default=1024)
parser.add_argument('--epoch', type=int, help='epoch size', default=300)
parser.add_argument('--epoch_counter', type=int, help='epoch counter', default=1)
parser.add_argument('--ch_encoder', type=int, help='number of latent channels', default=128)
parser.add_argument('--ch_decoder', type=int, help='number of latent channels', default=128)
parser.add_argument('--num_encoder_layers', type=int, help='number of encoder layers', default=6)
parser.add_argument('--num_decoder_layers', type=int, help='number of decoder layers', default=6)
parser.add_argument('--itr', type=int, help='number of training iterations', default=1)
parser.add_argument('--fourier_encoding', type=bool, help='includes fourier encoding of pos', default=True)
parser.add_argument('--parametric_encoding', type=bool, help='includes parametric encoding of pos', default=True)
parser.add_argument('--n_fourier_features', type=int, help='number of fourier features', default=12)
parser.add_argument('--coarse_resolution', type=tuple, help='tuple of coarse resolution for x, y, z', default=(16, 16, 16))
parser.add_argument('--fine_resolution', type=tuple, help='tuple of fine resolution for x, y, z', default=(256, 256, 256))
parser.add_argument('--num_levels', type=int, help='number of levels', default=6)
parser.add_argument('--num_nodes', type=int, help='number of nodes', default=2**30)
parser.add_argument('--num_channels', type=int, help='number of channels', default=12)
parser.add_argument('--dependence_method', type=str, default='pearson', help='statistical dependence method')
args_dict = vars(parser.parse_args())


# args_dict = {
#     'symm': 'mul', 
#     'lr': 0.0008, 
#     'batch': 1024,
#     'epoch': 300, 
#     'epoch_counter': 1,
#     'ch_encoder': 128,
#     'ch_decoder': 128,
#     'num_encoder_layers': 6,
#     'num_decoder_layers': 6,
#     'itr': 1,
#     'fourier_encoding': True,
#     'parametric_encoding': True,
#     'n_fourier_features': 12,
#     'coarse_resolution': 16,
#     'fine_resolution': 256,
#     'num_levels': 6,
#     'num_nodes': 30,
#     'num_channels': 12,
#     'dependence_method': 'pearson'
# }


SOURCE_PATH = ProjectConfigs().DATA_PATH
DESCRIPTION_FILE_NAME = ProjectConfigs().DESCRIPTION_PATH
EXPERIMENT_DIR = ProjectConfigs().EXPEREMENT_PATH
EXPERIMENT_PATH = os.path.abspath(EXPERIMENT_DIR)
USE_BATCH_LOADER = True
CONFIG_PATH = './configs'
train_dir = './sampled_data/train'
validation_dir = './sampled_data/validation'

with open(f'{CONFIG_PATH}/encoder_configs.json') as f:
    encoder_configs = json.load(f)

grid_channels = int(args_dict['num_channels'] // args_dict['num_levels'])
if args_dict['fourier_encoding']:
    in_channels = 2 * args_dict['n_fourier_features'] + 3
    if args_dict['parametric_encoding']:
        in_channels = grid_channels * args_dict['num_levels'] + 2 * args_dict['n_fourier_features'] + 3
else:
    in_channels = 3

channels_encoder = args_dict['ch_encoder']
channels_decoder = args_dict['ch_decoder']
batch_size = args_dict['batch']
learning_rates = args_dict['lr']
symm_method = args_dict['symm']
num_encoder_layers = args_dict['num_encoder_layers']
num_decoder_layers = args_dict['num_decoder_layers']
epochs = args_dict['epoch']
experiment_iter = args_dict['itr']
corr_method = args_dict['dependence_method']


if not os.path.isdir(EXPERIMENT_DIR):
    os.makedirs(EXPERIMENT_PATH)
experiment_folder = f"{EXPERIMENT_PATH}"
if not os.path.isdir(experiment_folder):
    os.makedirs(os.path.abspath(experiment_folder))
writer = SummaryWriter(f'{experiment_folder}/summary')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def main():
    validation_data = xr.open_mfdataset(f'{validation_dir}/*.nc')
    pos_ref_val = torch.from_numpy(validation_data.pos_ref.values)
    pos_val = torch.from_numpy(validation_data.pos.values)
    if corr_method == 'pearson':
        pearson_val = torch.from_numpy(validation_data.pearson.values)
        validation_input = torch.cat([pos_ref_val, pos_val, pearson_val], dim=1).to(device)
    elif corr_method == 'mi':
        mi_val = torch.from_numpy(validation_data.__xarray_dataarray_variable__.values)
        validation_input = torch.cat([pos_ref_val, pos_val, mi_val], dim=1).to(device)
    else:
        raise NotImplementedError()
    
    dataset_validation = SampledEnsembleDataset(validation_input)
    if USE_BATCH_LOADER:
        validation_loader = BatchLoader(dataset_validation, batch_size=batch_size)
    else:
        validation_loader = DataLoader(dataset_validation, batch_size=batch_size, num_workers=4)
    
    fourier_features = NerfFourierFeatures(3, args_dict['n_fourier_features']).to(device)
    feature_grid = tcnn.Encoding(3, encoder_configs["encoding"])

    input_processing = InputProcessingPosition(fourier_features, feature_grid)
    print(input_processing)
    
    if symm_method == "sum":
        symm = SymmNetAdd()
        bottelneck_ch = channels_encoder
    elif symm_method == "sum-diff":
        symm = SymmNetAddDiff()
        bottelneck_ch = 2 * channels_encoder
    elif symm_method == "mul":
        symm = Multiplier()
        bottelneck_ch = channels_encoder
    else:
        raise NotImplementedError()
    
    encoder = PytorchEncoder(in_channels, channels_encoder, num_encoder_layers)
    decoder = PytorchDecoder(bottelneck_ch, channels_decoder, 1, num_decoder_layers) 
    main_decoder = MainDecoderPosition(symm, decoder).to(device)
    encoder_decoder = EncoderDecoder(encoder, main_decoder)
    model = encoder_decoder.to(device)
    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rates)
    loss_function = nn.L1Loss()
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=20, threshold=0.001)
    stats_tracker = WelfordStatisticsTracker()
    relu = nn.ReLU()
    
    models_dir = f'{experiment_folder}/models'
    if not os.path.isdir(models_dir):
        os.makedirs(os.path.abspath(models_dir))
    
    n = 0
    for e in range(epochs):
        # Training loop
        model.train()
        stats_tracker.reset()
        if e % args_dict['epoch_counter'] == 0:
            train_data = xr.open_mfdataset(f'{train_dir}/*_train{n}.nc')
            n += 1
            pos_ref_train = torch.from_numpy(train_data.pos_ref.values)
            pos_train = torch.from_numpy(train_data.pos.values)
            if corr_method == 'pearson':
                pearson_train = torch.from_numpy(train_data.pearson.values)
                train_input = torch.cat([pos_ref_train, pos_train, pearson_train], dim=1).to(device)
            elif corr_method == 'mi':
                mi_train = torch.from_numpy(train_data.__xarray_dataarray_variable__.values)
                train_input = torch.cat([pos_ref_train, pos_train, mi_train], dim=1).to(device)
            else:
                raise NotImplementedError()
            dataset_train = SampledEnsembleDataset(train_input)
            if USE_BATCH_LOADER:
                train_loader = BatchLoader(dataset_train, batch_size=batch_size, shuffle=True)
            else:
                train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=4)
        print('[INFO] Starting epoch {}. {} batches to train.'.format(e, len(train_loader)))
        pbar = ProgressBar(len(dataset_train))

        for i, batch in enumerate(train_loader):
            positions_ref, positions, corr_true = batch
            corr_true = corr_true.unsqueeze(1)
            pos_ref = input_processing(positions_ref)
            pos = input_processing(positions)
            
            opt.zero_grad()
            f_ref = encoder(pos_ref)
            f = encoder(pos)
            out = main_decoder(f_ref, f)
            prediction = out
            loss = loss_function(prediction, corr_true)
            loss.backward()
            opt.step()
            loss = loss.item()
            stats_tracker.update(loss, weight=len(batch))
            pbar.step(batch[0].shape[0])

        writer.add_scalar(f"Loss/train_{args_dict['lr']}", stats_tracker.mean(), e)
        writer.flush()
        print('[INFO] Training loss after epoch {}: {}+-{}'.format(e, stats_tracker.mean(), stats_tracker.std()))

        # Validation loop
        model.eval()
        stats_tracker.reset()
        with torch.no_grad():
            pbar = ProgressBar(len(dataset_validation))
            for i, batch in enumerate(validation_loader):
                positions_ref, positions, corr_true = batch
                corr_true = corr_true.unsqueeze(1)
                pos_ref = input_processing(positions_ref)
                pos = input_processing(positions)
                
                opt.zero_grad()
                f_ref = encoder(pos_ref)
                f = encoder(pos)
                out = main_decoder(f_ref, f)
                if corr_method == 'pearson':
                    out = torch.clamp(out, min=-1, max=1)
                elif corr_method == 'mi':
                    out = relu(out)
                prediction = out
                loss = loss_function(prediction, corr_true)
                loss = loss.item()
                stats_tracker.update(loss, weight=len(batch))
                pbar.step(batch[0].shape[0])
        scheduler.step(loss)
        writer.add_scalar(f"Loss/test_{args_dict['lr']}", stats_tracker.mean(), e)
        writer.flush()
        print('[INFO] validation loss after epoch {}: {}+-{}'.format(e, stats_tracker.mean(), stats_tracker.std()))
        if (e + 1) % 50 == 0:
            torch.save(model.state_dict(), f'{models_dir}/mi_srn_{e + 1}.pth')
            torch.save(feature_grid.state_dict(), f'{models_dir}/input_processing{e + 1}.pth')
    torch.save(model.state_dict(), f'{models_dir}/mi_srn_{e + 1}.pth')
    torch.save(feature_grid.state_dict(), f'{models_dir}/input_processing{e + 1}.pth')
    torch.save(encoder.state_dict(), f'{models_dir}/encoder{e + 1}.pth')
    torch.save(main_decoder.state_dict(), f'{models_dir}/decoder{e + 1}.pth')
    shutil.copyfile(f'{CONFIG_PATH}/config_encoder.json', f'{models_dir}/config_encoder.json')
    shutil.copyfile(f'{CONFIG_PATH}/config_decoder.json', f'{models_dir}/config_decoder.json')
    with open(f'{models_dir}/config.json', 'w') as file:
        if symm_method == "sum":
            symm_name = 'Add'
        elif symm_method == "sum-diff":
            symm_name = 'AddDiff'
        elif symm_method == "mul":
            symm_name = 'Mul'
        else:
            raise NotImplementedError()
        if corr_method == 'pearson':
            is_mi = 'false'
        elif corr_method == 'mi':
            is_mi = 'true'
        else:
            raise NotImplementedError()
        file.write(
            '{\n' +
            f'    "network_type": "MINE_SRN",\n' +
            f'    "symmetrizer_type": "{symm_name}",\n' +
            f'    "is_mutual_information": {is_mi}\n' +
            '}\n')
    with open(DESCRIPTION_FILE_NAME, 'w') as fp:
        json.dump(args_dict, fp, indent=4)

if __name__ == '__main__':
    main()
