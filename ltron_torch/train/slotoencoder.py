#!/usr/bin/env python
import math
import os

import tqdm

import PIL.Image as Image

import numpy

import torch
from torch.utils.data import Dataset, DataLoader

from ltron.dataset.paths import get_dataset_paths

import ltron_torch.models.slotoencoder as slotoencoder
from ltron_torch.gym_tensor import (
    default_image_transform, default_image_untransform)

class FrameDataset(Dataset):
    def __init__(self, dataset, split, subset=None):
        dataset_paths = get_dataset_paths(dataset, split, subset)
        x_paths = dataset_paths['color_x']
        y_paths = dataset_paths['color_y']
        image_paths = [
            [x_path.replace('_0000.png', '_%04i.png'%i) for i in range(7)]
            for x_path in x_paths
        ]
        image_paths.append(y_paths)
        self.color_paths = numpy.concatenate(image_paths)

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, index):
        path = self.color_paths[index]
        image = Image.open(path)
        x = default_image_transform(image)
        y = torch.LongTensor(numpy.array(image)).permute(2,0,1)

        return x, y, path

def train(
    epochs=50,
    dataset_name='conditional_snap_two_frames',
    batch_size=4,
):
    
    train_dataset = FrameDataset(dataset_name, split='train')
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    test_dataset = FrameDataset(dataset_name, split='test')
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    config = 
    model = slotoencoder.SlotoEncoder(
        
