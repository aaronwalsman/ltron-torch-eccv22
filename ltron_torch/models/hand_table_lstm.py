import random
import os

import numpy

import torch
from torch.nn import Module, Linear, Embedding, LSTM, Dropout
from torch.distributions import Categorical

from ltron.config import Config
from ltron.dataset.paths import get_dataset_info

from ltron_torch.gym_tensor import default_image_transform
from ltron_torch.models.heads import (
    LinearMultiheadDecoder, Conv2dMultiheadDecoder)

from ltron_torch.models.resnet import named_backbone, named_encoder_channels
from ltron_torch.models.mlp import conv2d_stack
from ltron_torch.models.simple_fcn import SimpleDecoder

from ltron_torch.models.deeplabv3 import resnet
from ltron_torch.models.deeplabv3.feature_extraction import create_feature_extractor

# build functions ==============================================================

class HandTableLSTMConfig(Config):
    num_modes = 6
    num_shapes = 6
    num_colors = 6
    table_channels = 2
    hand_channels = 2
    
    resnet_backbone = 'resnet18'
    pretrain_resnet = True
    freeze_resnet = False
    compact_visual_channels = 64
    global_table_channels = 768
    global_hand_channels = 768
    reassembly_channels = 512
    lstm_hidden_channels = 512
    decoder_channels = 256
    visual_dropout = 0.
    global_dropout = 0.
    table_shape = (8,8)
    hand_shape = (3,3)

    pretrained_fcn_path = None

class HandTableLSTM(Module):

    def load_fcn_backbone(self):
        backbone = resnet.resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True])
        backbone = create_feature_extractor(backbone, {"layer4": "out"})
        pretrained_fcn = torch.load(os.path.expanduser(
            "/gscratch/raivn/muruz/ltron-torch/ltron_torch/train/checkpoint/Jan29_08-37-01_patillo/model_0005.pt"))
        model_dict = backbone.state_dict()
        filtered_dict = {k[9:]:v for k, v in pretrained_fcn.items() if k[9:] in model_dict}
        model_dict.update(filtered_dict)
        return backbone

    def __init__(self, config, checkpoint=None):
        # intiialization and storage
        super().__init__()
        self.config = config
        
        # resnet backbone
        fcn_layers = ('layer4', 'layer3', 'layer2', 'layer1')
        self.visual_backbone = named_backbone(
            config.resnet_backbone,
            *fcn_layers,
            frozen_batchnorm=True,
            pretrained=config.pretrain_resnet,
            frozen_weights=config.freeze_resnet,
        )
        if config.pretrained_fcn_path:
            pretrained_fcn = torch.load(os.path.expanduser(config.pretrained_fcn_path))
            model_dict = self.visual_backbone.state_dict()
            filtered_dict = { k[8:]:v for k, v in pretrained_fcn.items() if k[8:] in model_dict}
            print(filtered_dict.keys())
            model_dict.update(filtered_dict)

        # self.visual_backbone = self.load_fcn_backbone()
        
        # visual feature extractor
        resnet_channels = named_encoder_channels(config.resnet_backbone)
        self.visual_stack = conv2d_stack(
            num_layers=2,
            in_channels=resnet_channels[0],
            hidden_channels=resnet_channels[0]//2,
            out_channels=config.compact_visual_channels,
        )

        # import pdb
        # pdb.set_trace()
        
        # visual dropout
        self.visual_dropout = Dropout(config.visual_dropout)
        
        # global feature layers
        table_pixels = config.table_shape[0] * config.table_shape[1]
        table_channels = config.compact_visual_channels * table_pixels
        self.table_feature = Linear(
            table_channels, config.global_table_channels)
        hand_pixels = config.hand_shape[0] * config.hand_shape[1]
        hand_channels = config.compact_visual_channels * hand_pixels
        self.hand_feature = Linear(
            hand_channels, config.global_hand_channels)
        
        # reassembly embedding
        self.reassembly_embedding = Embedding(2, config.reassembly_channels)
        
        # global dropout
        self.global_dropout = Dropout(config.global_dropout)
        
        # build the lstm
        lstm_in_channels = (
            config.global_table_channels +
            config.global_hand_channels +
            config.reassembly_channels
        )
        self.lstm = LSTM(
            input_size=lstm_in_channels,
            hidden_size=config.lstm_hidden_channels,
            num_layers=1,
        )
        
        # map decoders
        self.dense_linear = Linear(
            config.lstm_hidden_channels, resnet_channels[0])
        
        self.table_decoder = SimpleDecoder(
            encoder_channels=resnet_channels,
            decoder_channels=config.decoder_channels,
        )
        self.dense_table_heads = Conv2dMultiheadDecoder(
            config.decoder_channels, config.table_channels, kernel_size=1)
        
        self.hand_decoder = SimpleDecoder(
            encoder_channels=resnet_channels,
            decoder_channels=config.decoder_channels,
        )
        self.dense_hand_heads = Conv2dMultiheadDecoder(
            config.decoder_channels, config.hand_channels, kernel_size=1)
        
        # global heads
        global_head_spec = {
            'mode':config.num_modes,
            'shape':config.num_shapes,
            'color':config.num_colors,
        }
        self.global_heads = LinearMultiheadDecoder(
            config.lstm_hidden_channels,
            global_head_spec,
        )
        
        if checkpoint is not None:
            if isinstance(checkpoint, str):
                checkpoint = torch.load(checkpoint)
            self.load_state_dict(checkpoint)
    
    def initialize_memory(self, batch_size):
        device = next(self.parameters()).device
        hidden_state = torch.zeros(
            1, batch_size, self.config.lstm_hidden_channels, device=device)
        cell_state = torch.zeros(
            1, batch_size, self.config.lstm_hidden_channels, device=device)
        return hidden_state, cell_state
    
    def reset_memory(self, memory, terminal):
        hidden, cell = memory
        for i, t in enumerate(terminal):
            if t:
                hidden[:,i] = 0
                cell[:,i] = 0
    
    def forward(self, x_table, x_hand, r, memory=None):
        
        # compute visual features
        x_layers = []
        xs = []
        for x in x_table, x_hand:
            s, b, c, h, w = x.shape
            x_layer = self.visual_backbone(x.view(s*b, c, h, w))
            x_layers.append(x_layer)
            # import pdb
            # pdb.set_trace()
            x = self.visual_stack(x_layer[0])
            sb, c, h, w = x.shape
            xs.append(x.view(s, b, c, h, w))
        
        x_table, x_hand = xs
        x_table_layers, x_hand_layers = x_layers
        
        # compute global feature
        x_table = x_table.view(s, b, -1)
        x_table = self.table_feature(x_table)
        
        x_hand = x_hand.view(s, b, -1)
        x_hand = self.hand_feature(x_hand)
        
        xr = self.reassembly_embedding(r)
        
        x_global = torch.cat((x_table, x_hand, xr), dim=-1)
        
        # compute sequence features
        x, hcn = self.lstm(x_global, memory)
        s, b, c = x.shape
        
        # compute global action features
        x_out = self.global_heads(x)
        x_out['memory'] = hcn
        
        # compute maps
        dense_x = self.dense_linear(x).view(s*b, -1, 1, 1)
        
        layer_4w = x_table_layers[0] + dense_x
        xw_dense = self.table_decoder(layer_4w, *x_table_layers[1:])
        xw_dense = self.dense_table_heads(xw_dense)
        sh, c, h, w = xw_dense.shape
        xw_dense = xw_dense.view(s, b, c, h, w)
        x_out['table'] = xw_dense
        
        layer_4h = x_hand_layers[0] + dense_x
        xh_dense = self.hand_decoder(layer_4h, *x_hand_layers[1:])
        xh_dense = self.dense_hand_heads(xh_dense)
        sh, c, h, w = xh_dense.shape
        xh_dense = xh_dense.view(s, b, c, h, w)
        x_out['hand'] = xh_dense
        
        return x_out
