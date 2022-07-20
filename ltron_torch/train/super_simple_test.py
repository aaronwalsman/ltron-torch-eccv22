#!/usr/bin/env python
import random
import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader

import numpy

from PIL import Image

import tqdm

from ltron.visualization.drawing import block_upscale_image

from ltron_torch.models.positional_encoding import positional_encoding

class SimpleModel(torch.nn.Module):
    def __init__(
        self,
        channels=256,
    ):
        super(SimpleModel, self).__init__()
        self.register_buffer(
            'sequence_encoding',
            positional_encoding(channels, 5000).unsqueeze(1)
        )
        
        self.in_embedding = torch.nn.Embedding(3, channels)
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            4,
            channels,
            0.1,
        )
        self.transformer_1 = torch.nn.TransformerEncoder(
            transformer_layer, 6)
        
        self.mode_linear = torch.nn.Linear(channels, 2)
    
    def forward(self, x):
        
        b = x.shape[1]
        x = self.in_embedding(x)
        
        out = torch.LongTensor([2]).view(1,1).expand(1,b).to(x.device)
        out = self.in_embedding(out)
        
        out_x = torch.cat((out, x), dim=0)
        s = out_x.shape[0]
        out_x = out_x + self.sequence_encoding[:s]
        
        out_x = self.transformer_1(out_x)
        
        out = out_x[[0]]
        
        mode = self.mode_linear(out[0])
        
        return mode

def train():
    model = SimpleModel().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    
    for epoch in range(1, 50):
        print('epoch: %i'%epoch)
        train_epoch(model, optimizer)

def train_epoch(model, optimizer):
    print('train')
    model.train()
    iterate = tqdm.tqdm(range(1000))
    for i in iterate:
        r = torch.LongTensor([random.randint(0,1) for _ in range(64)]).cuda()
        x = r.unsqueeze(0)
        
        total_loss = 0.
        
        mode = model(x)
        
        mode_loss = torch.nn.functional.cross_entropy(mode, r)
        total_loss = total_loss + mode_loss
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        mode_max = torch.argmax(mode, dim=-1)
        mode_correct = mode_max == r
        b = mode_correct.shape[0]
        mode_accuracy = torch.sum(mode_correct).float() / b
        
        iterate.set_description('acc: %.02f'%float(mode_accuracy))

if __name__ == '__main__':
    train()
