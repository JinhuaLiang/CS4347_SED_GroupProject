import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from pytorch_utils import do_mixup


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(Block, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels)
        
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv)
        init_bn(self.bn)
        
    def forward(self, input):
        
        x = input
        x = self.bn(self.conv(x))
        x = F.glu(x, dim=1) # (batch_size, channels, time_steps, mel_bins)
        
        return x


class AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation='linear', temperature=1.):
        super(AttBlock, self).__init__()
        
        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.bn_att = nn.BatchNorm1d(n_out)
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)
        
    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        tmp = self.att(x)
        tmp = torch.clamp(tmp, -10, 10)
        att = torch.exp(tmp / self.temperature) + 1e-6
        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


# The following CRNN architecture are designed following Yong Xu's code:
# https://github.com/yongxuUSTC/dcase2017_task4_cvssp
class Cnn_Gru(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn_Gru, self).__init__()
        
        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=16, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
        
        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.block1 = Block(in_channels=1, out_channels=128)
        self.block2 = Block(in_channels=64, out_channels=128)
        self.block3 = Block(in_channels=64, out_channels=128)
        self.block4 = Block(in_channels=64, out_channels=128)
        self.block5 = Block(in_channels=64, out_channels=128)
        self.block6 = Block(in_channels=64, out_channels=128)
        self.block7 = Block(in_channels=64, out_channels=128)
        self.block8 = Block(in_channels=64, out_channels=128)

        self.conv9 = nn.Conv2d(in_channels=64, 
                               out_channels=256,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=True)

        self.bigru = nn.GRU(input_size=256, hidden_size=128, num_layers=1, 
            bias=True, batch_first=True, bidirectional=True)

        self.bigru_g = nn.GRU(input_size=256, hidden_size=128, num_layers=1, 
            bias=True, batch_first=True, bidirectional=True)
        
        self.att_block = AttBlock(n_in=256, n_out=classes_num, activation='sigmoid')

        self.init_weights()
    
    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.conv9)
        init_gru(self.bigru)
        init_gru(self.bigru_g)

    def forward(self, input, mixup_lambda=None):
        
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
    
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.block1(x)
        x = self.block2(x)
        x = F.max_pool2d(x, kernel_size=(1, 2))
        
        x = self.block3(x)
        x = self.block4(x)
        x = F.max_pool2d(x, kernel_size=(1, 2))
        
        x = self.block5(x)
        x = self.block6(x)
        x = F.max_pool2d(x, kernel_size=(1, 2))
        
        x = self.block7(x)
        x = self.block8(x)
        x = F.max_pool2d(x, kernel_size=(1, 2))
        
        x = F.relu_(self.conv9(x))
        (x, _) = torch.max(x, dim=3)
        
        x = x.transpose(1, 2) # (batch_size, time_steps, channels)
        (rnnout, _) = self.bigru(x)
        (rnnout_gate, _) = self.bigru_g(x)
        
        x = rnnout.transpose(1, 2) * rnnout_gate.transpose(1, 2)
        """x.shape = (batch_size, channels, time_steps)"""
        
        (clipwise_output, norm_att, cla) = self.att_block(x)
        """cla.shape = (batch_size, classes_num, time_steps)"""
        
        output_dict = {
            'framewise_output': cla.transpose(1, 2), 
            'clipwise_output': clipwise_output, 
            'embedding': cla}
        
        return output_dict