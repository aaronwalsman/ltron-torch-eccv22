'''
adapted from:
https://github.com/openai/DALL-E/blob/master/dall_e/encoder.py
'''
from collections import OrderedDict

import torch
from torch.nn import (
    Module, Identity, Conv2d, Sequential, ReLU, MaxPool2d, Upsample)
from torch.nn.functional import gumbel_softmax, one_hot

class EncoderBlockA(Module):
    '''
    Resnet Bottleneck
    '''
    def __init__(self, in_channels, out_channels, num_layers):
        super(EncoderBlockA, self).__init__()
        
        hidden_channels = out_channels // 4
        self.post_gain = 1. / num_layers**2
        
        if in_channels == out_channels:
            self.identity_path = Identity()
        else:
            self.identity_path = Conv2d(
                in_channels, out_channels, kernel_size=1)
        
        self.residual_path = Sequential(OrderedDict([
            ('relu_1', ReLU()),
            ('conv_1', Conv2d(
                in_channels, hidden_channels, kernel_size=1)),
            ('relu_2', ReLU()),
            ('conv_2', Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, padding=1)),
            ('relu_3', ReLU()),
            ('conv_3', Conv2d(
                hidden_channels, out_channels, kernel_size=1)),
        ]))
    
    def forward(self, x):
        return self.identity_path(x) + self.post_gain * self.residual_path(x)

class EncoderBlockB(Module):
    '''
    Dalle DVAE
    '''
    def __init__(self, in_channels, out_channels, num_layers):
        super(EncoderBlockB, self).__init__()
        
        hidden_channels = out_channels // 4
        self.post_gain = 1. / num_layers**2
        
        if in_channels == out_channels:
            self.identity_path = Identity()
        else:
            self.identity_path = Conv2d(
                in_channels, out_channels, kernel_size=1)
        
        self.residual_path = Sequential(OrderedDict([
            ('relu_1', ReLU()),
            ('conv_1', Conv2d(
                in_channels, hidden_channels, kernel_size=3, padding=1)),
            ('relu_2', ReLU()),
            ('conv_2', Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, padding=1)),
            ('relu_3', ReLU()),
            ('conv_3', Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, padding=1)),
            ('relu_4', ReLU()),
            ('conv_4', Conv2d(
                hidden_channels, out_channels, kernel_size=1)),
        ]))
    
    def forward(self, x):
        return self.identity_path(x) + self.post_gain * self.residual_path(x)

class EncoderGroup(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks,
        num_layers,
        downsample,
        EncoderBlock=EncoderBlockB
    ):
        super(EncoderGroup, self).__init__()
        
        assert num_blocks >= 1
        blocks = OrderedDict()
        blocks['block_1'] = EncoderBlock(in_channels, out_channels, num_layers)
        for i in range(1, num_blocks):
            blocks['block_%i'%(i+1)] = EncoderBlock(
                out_channels, out_channels, num_layers)
        if downsample:
            blocks['downsample'] = MaxPool2d(kernel_size=2)
        
        self.blocks = Sequential(blocks)
    
    def forward(self, x):
        return self.blocks(x)

class Encoder(Module):
    def __init__(
        self,
        in_channels=3,
        hidden_channels=256,
        vocabulary_size=8192,
        blocks_per_group=2,
    ):
        super(Encoder, self).__init__()
        
        num_groups = 4
        num_layers = num_groups * blocks_per_group
        
        groups = OrderedDict()
        groups['in'] = Conv2d(
            in_channels, hidden_channels, kernel_size=7, padding=3)
        groups['group_1'] = EncoderGroup(
            hidden_channels,
            hidden_channels,
            blocks_per_group,
            num_layers,
            downsample=True,
        )
        groups['group_2'] = EncoderGroup(
            hidden_channels,
            hidden_channels*2,
            blocks_per_group,
            num_layers,
            downsample=True,
        )
        groups['group_3'] = EncoderGroup(
            hidden_channels*2,
            hidden_channels*4,
            blocks_per_group,
            num_layers,
            downsample=True,
        )
        groups['group_4'] = EncoderGroup(
            hidden_channels*4,
            hidden_channels*8,
            blocks_per_group,
            num_layers,
            downsample=False,
        )
        groups['output'] = Sequential(OrderedDict([
            ('relu', ReLU()),
            ('conv', Conv2d(8*hidden_channels, vocabulary_size, 1)),
        ]))
        
        self.groups = Sequential(groups)
    
    def forward(self, x):
        return self.groups(x)

class DecoderBlock(Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(DecoderBlock, self).__init__()
        
        hidden_channels = out_channels // 4
        self.post_gain = 1. / num_layers**2
        
        if in_channels == out_channels:
            self.identity_path = Identity()
        else:
            self.identity_path = Conv2d(
                in_channels, out_channels, kernel_size=1)
        
        self.residual_path = Sequential(OrderedDict([
            ('relu_1', ReLU()),
            ('conv_1', Conv2d(
                in_channels, hidden_channels, kernel_size=1)),
            ('relu_2', ReLU()),
            ('conv_2', Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, padding=1)),
            ('relu_3', ReLU()),
            ('conv_3', Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, padding=1)),
            ('relu_4', ReLU()),
            ('conv_4', Conv2d(
                hidden_channels, out_channels, kernel_size=3, padding=1)),
        ]))
    
    def forward(self, x):
        return self.identity_path(x) + self.post_gain * self.residual_path(x)

class DecoderGroup(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks,
        num_layers,
        upsample
    ):
        super(DecoderGroup, self).__init__()
        
        assert(num_blocks >= 1)
        blocks = OrderedDict()
        blocks['block_1'] = DecoderBlock(in_channels, out_channels, num_layers)
        for i in range(1, num_blocks):
            blocks['block_%i'%(i+1)] = DecoderBlock(
                out_channels, out_channels, num_layers)
        if upsample:
            blocks['upsample'] = Upsample(scale_factor=2, mode='nearest')
        
        self.blocks = torch.nn.Sequential(blocks)
    
    def forward(self, x):
        return self.blocks(x)

class Decoder(Module):
    def __init__(
        self,
        initial_channels=128,
        hidden_channels=256,
        output_channels=3,
        vocabulary_size=8192,
        blocks_per_group=2,
    ):
        super(Decoder, self).__init__()
        
        num_groups = 4
        num_layers = num_groups * blocks_per_group
        
        groups = OrderedDict()
        groups['in'] = Conv2d(
            vocabulary_size, initial_channels, kernel_size=1)
        groups['group_1'] = DecoderGroup(
            initial_channels,
            hidden_channels*8,
            blocks_per_group,
            num_layers,
            upsample=True,
        )
        groups['group_2'] = DecoderGroup(
            hidden_channels*8,
            hidden_channels*4,
            blocks_per_group,
            num_layers,
            upsample=True,
        )
        groups['group_3'] = DecoderGroup(
            hidden_channels*4,
            hidden_channels*2,
            blocks_per_group,
            num_layers,
            upsample=True,
        )
        groups['group_4'] = DecoderGroup(
            hidden_channels*2,
            hidden_channels,
            blocks_per_group,
            num_layers,
            upsample=False
        )
        groups['output'] = Sequential(OrderedDict([
            ('relu', ReLU()),
            ('conv', Conv2d(
                hidden_channels, output_channels, kernel_size=1)),
        ]))
        
        self.groups = torch.nn.Sequential(groups)
    
    def forward(self, x):
        return self.groups(x)

class TokenSampler(Module):
    def __init__(self, vocabulary_size, tau=1., dim=-3):
        super(TokenSampler, self).__init__()
        
        self.vocabulary_size = vocabulary_size
        self.tau = tau
        self.dim = dim
    
    def forward(self, z):
        if self.training:
            z = gumbel_softmax(z, self.tau, dim=self.dim)
        else:
            z = torch.argmax(z, dim=self.dim)
            z = one_hot(z, num_classes=self.vocabulary_size)
            z = z.permute(0, 3, 1, 2).float()
        return z

class DVAE(Module):
    def __init__(
        self,
        in_channels=3,
        hidden_channels=256,
        initial_decoder_channels=128,
        output_channels=3,
        vocabulary_size=8192,
        blocks_per_group=2,
        tau=1.
    ):
        super(DVAE, self).__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            vocabulary_size=vocabulary_size,
            blocks_per_group=blocks_per_group,
        )
        self.sampler = TokenSampler(
            vocabulary_size=vocabulary_size,
            tau=tau,
            dim=-3,
        )
        self.decoder = Decoder(
            initial_channels=initial_decoder_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            vocabulary_size=vocabulary_size,
            blocks_per_group=blocks_per_group,
        )
    
    def update_tau(self, tau):
        self.sampler.tau = tau
    
    def forward(self, x, sample_only=False):
        z = self.encoder(x)
        z_sample = self.sampler(z)
        if sample_only:
            return z_sample
        else:
            x = self.decoder(z_sample)
            return z, z_sample, x
