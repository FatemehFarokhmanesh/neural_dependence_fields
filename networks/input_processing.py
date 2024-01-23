import torch
from torch import nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class InputProcessing(nn.Module):

    def __init__(self, fourier_module, 
                        latent_feature_module, members, fourier_encoding=True, parametric_encoding= True):
        super(InputProcessing, self).__init__()
        self.fourier_module = fourier_module
        self.latent_feature_module = latent_feature_module
        self.members = members
        self.fourier_encoding = fourier_encoding
        self.parametric_encoding = parametric_encoding

    def forward(self, inputs, pos):
        positions = pos.unsqueeze(1).repeat(1, self.members, 1)
        inputs = inputs.unsqueeze(-1)
        if self.fourier_encoding:
            if self.parametric_encoding:
                encoding = self.fourier_module(pos)
                encoding = encoding.unsqueeze(1).repeat(1, self.members, 1)
                trainable_encoding = self.latent_feature_module(pos)
                trainable_encoding = trainable_encoding.unsqueeze(1).repeat(1, self.members, 1)
                inputs_list = [trainable_encoding, encoding.to(device), inputs, positions]
            else:
                encoding = self.fourier_module(pos)
                encoding = encoding.unsqueeze(1).repeat(1, self.members, 1)
                inputs_list = [encoding, inputs, positions]
        else:
            inputs_list = [inputs, positions]
        inputs = torch.cat(inputs_list, dim=-1)
        return inputs

class InputProcessingPosition(nn.Module):

    def __init__(self, fourier_module, 
                        latent_feature_module, fourier_encoding=True, parametric_encoding= True):
        super(InputProcessingPosition, self).__init__()
        self.fourier_module = fourier_module
        self.latent_feature_module = latent_feature_module
        self.fourier_encoding = fourier_encoding
        self.parametric_encoding = parametric_encoding

    def forward(self, pos):
        
        if self.fourier_encoding:
            if self.parametric_encoding:
                encoding = self.fourier_module(pos)
                trainable_encoding = self.latent_feature_module(pos)
                inputs_list = [pos, encoding.to(device), trainable_encoding]
                inputs = torch.cat(inputs_list, dim=-1)
            else:
                encoding = self.fourier_module(pos)
                inputs_list = [pos, encoding.to(device)]
                inputs = torch.cat(inputs_list, dim=-1)
        else:
            inputs = pos        
        
        return inputs

class InputProcessingPositionfou(nn.Module):

    def __init__(self, fourier_module, 
                        fourier_encoding=True):
        super(InputProcessingPositionfou, self).__init__()
        self.fourier_module = fourier_module
        self.fourier_encoding = fourier_encoding

    def forward(self, pos):
        
        if self.fourier_encoding:
                encoding = self.fourier_module(pos)
                # inputs_list = [pos, encoding.to('cuda')]
                inputs_list = [pos, encoding]
                inputs = torch.cat(inputs_list, dim=-1)
        else:
            inputs = pos        
        
        return inputs