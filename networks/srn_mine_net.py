import math
import json
import numpy as np
import torch
from torch import nn
import tinycudann as tcnn
from networks.activations import SnakeAlt


class PytorchEncoderGrid(nn.Module):
    def __init__(self, input_processing_module, in_channels, hidden_channels, num_layers,
                 activation='use_snake'):
        super(PytorchEncoderGrid, self).__init__()
        self.input_processing_module = input_processing_module
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.activation = activation
        if self.activation == 'use_snake':
            self.activation = SnakeAlt()
        else:
            self.activation = nn.ReLU()
        encoder = []
        encoder += [nn.Linear(self.in_channels, self.hidden_channels, bias=False),
            self.activation]
        for i in range(self.num_layers - 1):
            encoder += [nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            self.activation]
        self.encoder = nn.Sequential(*encoder)

    def forward(self, inputs, pos):
        x = self.input_processing_module(inputs, pos)
        out = self.encoder(x)
        return out

class PytorchEncoderPosition(nn.Module):
    def __init__(self, input_processing_module, in_channels, hidden_channels, num_layers,
                 activation='use_snake'):
        super(PytorchEncoderPosition, self).__init__()
        self.input_processing_module = input_processing_module
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.activation = activation
        if self.activation == 'use_snake':
            self.activation = SnakeAlt()
        else:
            self.activation = nn.ReLU()
        encoder = []
        encoder += [nn.Linear(self.in_channels, self.hidden_channels, bias=False),
            self.activation]
        for i in range(self.num_layers - 1):
            encoder += [nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            self.activation]
        self.encoder = nn.Sequential(*encoder)

    def forward(self, pos):
        x = self.input_processing_module(pos)
        out = self.encoder(x)
        return out

class PytorchEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 activation='use_snake'):
        super(PytorchEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.activation = activation
        if self.activation == 'use_snake':
            self.activation = SnakeAlt()
        else:
            self.activation = nn.ReLU()
        encoder = []
        encoder += [nn.Linear(self.in_channels, self.hidden_channels, bias=False),
            self.activation]
        for i in range(self.num_layers - 1):
            encoder += [nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            self.activation]
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        out = self.encoder(x)
        return out

class PytorchDecoder(nn.Module):
    def __init__(self, botteleneck_channels, hidden_channels, out_channels, num_layers, activation='use_snake'):
        super(PytorchDecoder, self).__init__()
        self.bottleneck_channels = botteleneck_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.activation = None
        if activation == 'use_snake':
            self.activation = SnakeAlt()
        else:
            self.activation = nn.ReLU()
        decoder = []
        decoder += [nn.Linear(self.bottleneck_channels, self.hidden_channels, bias=False),
            self.activation]
        for i in range(self.num_layers - 2):
            decoder += [nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            self.activation,]
        decoder += [nn.Linear(self.hidden_channels, self.out_channels, bias=False)]
        self.decoder = nn.Sequential(*decoder)
        
    def forward(self, x):
        out = self.decoder(x)
        return  out
        
class Encoder(nn.Module):
    def __init__(self, encoder_module):
        super(Encoder, self).__init__()
        self.encoder_module = encoder_module

    def forward(self, x):
        out = self.encoder_module(x)
        return  out

class SymmNetAddDiff(nn.Module):
    
    def forward(self, f_ref, f):
        sum = f_ref + f
        diff = abs(f_ref - f)
        out = torch.cat([sum, diff], dim=-1)
        return out

class SymmNetAdd(nn.Module):
    
    def forward(self, f_ref, f):
        return f_ref + f


class Decoder(nn.Module):
    def __init__(self, decoder_module):
        super(Decoder, self).__init__()
        self.decoder_module = decoder_module

    def forward(self, x):
        out = self.decoder_module(x)
        return  out

# Mine

class MineSummary(nn.Module):

    def forward(self, f_join, f_marg):
        t = torch.mean(f_join, dim=1)
        t_marg = torch.logsumexp(f_marg, dim=1) - torch.log(torch.full_like(t, f_marg.shape[1]))
        return - t + t_marg

class MainDecoder(nn.Module):
    def __init__(self, symm_module, decoder_module):
        super(MainDecoder, self).__init__()
        self.symm_module = symm_module
        self.decoder_module = decoder_module
        self.mine_module = MineSummary()

    def forward(self, f_ref, f):
        indices = torch.argsort(torch.rand((f.shape[0], f.shape[1]), device=f.device), dim=1)
        f_marg = f[torch.arange(f.shape[0]).unsqueeze(-1), indices]
        out_join = self.symm_module(f_ref, f)
        out_marg = self.symm_module(f_ref, f_marg)
        out_join = self.decoder_module(out_join)
        out_marg = self.decoder_module(out_marg)
        out = self.mine_module(out_join, out_marg)
        return out
    
class MainDecoderPosition(nn.Module):
    def __init__(self, symm_module, decoder_module):
        super(MainDecoderPosition, self).__init__()
        self.symm_module = symm_module
        self.decoder_module = decoder_module
        

    def forward(self, f_ref, f):
        out_join = self.symm_module(f_ref, f)
        out = self.decoder_module(out_join)
        return out

class Concatenator(nn.Module):
    
    def forward(self, f_t, f_z):
        out = torch.cat([f_t, f_z], dim=-1)
        return out

class Multiplier(nn.Module):
    def forward(self, f_t, f_z):
        out = f_t * f_z
        return out

class MainDecoderMultiVar(nn.Module):
    def __init__(self, cat_module, decoder_module):
        super(MainDecoderMultiVar, self).__init__()
        self.cat_module = cat_module
        self.decoder_module = decoder_module
        

    def forward(self, f_ref, f):
        out_join = self.cat_module(f_ref, f)
        out = self.decoder_module(out_join)
        return out

class EncoderDecoder(nn.Module):
    def __init__(self, encoder_module, decoder_module):
        super(EncoderDecoder, self).__init__()
        self.encoder_module = encoder_module
        self.decoder_module = decoder_module
    
    def forward(self, x, y):
        f_ref = self.encoder_module(x)
        f = self.encoder_module(y)
        out = self.decoder_module(f_ref, f)
        return out

class EncoderDecoderMultiVar(nn.Module):
    def __init__(self, encoder_var1, encoder_var2, decoder_module):
        super(EncoderDecoderMultiVar, self).__init__()
        self.encoder_var1 = encoder_var1
        self.encoder_var2 = encoder_var2
        self.decoder_module = decoder_module
    
    def forward(self, x, y):
        f_ref = self.encoder_var1(x)
        f = self.encoder_var2(y)
        out = self.decoder_module(f_ref, f)
        return out

class EncoderDecoderGrid(nn.Module):
    def __init__(self, encoder_module, decoder_module):
        super(EncoderDecoderGrid, self).__init__()
        self.encoder_module = encoder_module
        self.decoder_module = decoder_module
    
    def forward(self, inputs, positions):
        f_ref = self.encoder_module(inputs, positions)
        f = f_ref[torch.randperm(f_ref.shape[0])]
        out = self.decoder_module(f_ref, f)
        return out


class EncoderDecoderValidation(nn.Module):
    def __init__(self, encoder_module, decoder_module):
        super(EncoderDecoderValidation, self).__init__()
        self.encoder_module = encoder_module
        self.decoder_module = decoder_module
    
    def forward(self, in_ref, pos_ref, in_, pos_):
        f_ref = self.encoder_module(in_ref, pos_ref)
        f = self.encoder_module(in_, pos_)
        out = self.decoder_module(f_ref, f)
        return out

class MineSRNet(nn.Module):
    def __init__(self, encoder_decoder_module):
        super(MineSRNet, self).__init__()
        self.running_mean = 0
        self.mine_module = MineSummary()
        self.encoder_decoder_module = encoder_decoder_module

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            t, t_marg = self._forward_without_marginal(x, z)
        else:
            t, t_marg = self._forward_with_marginal(x, z, z_marg)
        out = self.mine_module(t, t_marg)
        return out

    def _forward_without_marginal(self, x, z):
        encoder = self.encoder_decoder_module.encoder_module
        decoder = self.encoder_decoder_module.decoder_module
        symmnet = self.encoder_decoder_module.symm_module
        f_ref = encoder(x)
        f = encoder(z)
        indices = torch.argsort(torch.rand((x.shape[0], x.shape[1]), device=x.device), dim=1)
        f_marg = f[torch.arange(x.shape[0]).unsqueeze(-1), indices]
        t = decoder(symmnet(f_ref, f))
        t_marg = decoder(symmnet(f_ref, f_marg))
        return t, t_marg

    def _forward_with_marginal(self, x, z, z_marg):
        t = torch.mean(self.encoder_decoder_module(x, z), dim=1)
        t_marg = self.encoder_decoder_module(x, z_marg)
        return t, t_marg

