#!/usr/bin/env python
import numpy

import torch
from torch.utils.data import Dataset, DataLoader

import tqdm

import PIL.Image as Image

import ltron.dataset.paths as paths
from ltron.visualization.drawing import draw_vector_field

from ltron_torch.gym_tensor import (
    default_image_transform, default_image_untransform)
from ltron_torch.models.simple_fcn import SimpleFCN
from ltron_torch.models.step_model import StepModel
from ltron_torch.models.mlp import Conv2dStack

class FrameDataset(Dataset):
    def __init__(
        self,
        dataset,
        split,
        subset=None,
    ):
        self.color_paths = paths.get_dataset_paths(dataset, split, subset)
    
    def __len__(self):
        return len(self.color_paths)
    
    def __getitem__(self, index):
        color_path = self.color_paths[index]
        color_image = Image.open(color_path)
        color = default_image_transform(color_image)
        
        pick_path = color_path.replace(
            'color_', 'pick_').replace('.png', '.npy')
        pick = torch.BoolTensor(numpy.load(pick_path)).unsqueeze(0)
        
        place_path = pick_path.replace('pick_', 'place_')
        place = torch.BoolTensor(numpy.load(place_path)).unsqueeze(0)
        
        offset_path = color_path.replace(
            'color_', 'offset_').replace('.png', '.npy')
        offset = numpy.load(offset_path)
        offset = numpy.moveaxis(offset, -1, 0)
        offset = torch.FloatTensor(offset)
        
        return color, pick, place, offset

def train(
    epochs = 10,
    batch_size=16,
    train_subset=None,
    test_subset=None,
):
    
    fcn = SimpleFCN(
        pretrained=True,
        compute_single=False,
    )
    dense_heads = {
        'pick' : Conv2dStack(3, 256, 256, 1),
        'place' : Conv2dStack(3, 256, 256, 1),
    }
         
    model = StepModel(fcn, dense_heads=dense_heads, single_heads={}).cuda()
    
    train_dataset = FrameDataset(
        'snap_one_frames',
        'train',
        subset=train_subset,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    
    test_dataset = FrameDataset(
        'snap_one_frames',
        'test',
        subset=test_subset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for epoch in range(1, epochs+1):
        iterate = tqdm.tqdm(train_loader)
        for color, pick, place, offsets in iterate:
            color = color.cuda()
            pick = pick.cuda()
            place = place.cuda()
            offsets = offsets.cuda()
            
            features = model(color)
            
            pick_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                features['pick'], pick.float())
            
            place_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                features['place'], place.float())
            
            iterate.set_description('pick: %.04f, place: %.04f'%(
                float(pick_loss), float(place_loss)))
            
            total_loss = pick_loss + place_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        torch.save(model.state_dict(), './model_%04i.pt'%epoch)
        
        with torch.no_grad():
            iterate = tqdm.tqdm(test_loader)
            total_pick_correct = 0
            total_place_correct = 0
            total_both_correct = 0
            total = 0
            for i, (color, pick, place, offsets) in enumerate(iterate):
                color = color.cuda()
                pick = pick.cuda()
                place = place.cuda()
                offsets = offsets.cuda()
                
                features = model(color)
                
                pick_prediction = features['pick']
                b, c, h, w = pick_prediction.shape
                pick_prediction = pick_prediction.view(b, h*w)
                pick_values, pick_indices = torch.max(
                    pick_prediction, dim=-1)
                pick_target = pick.view(b, h*w)[range(b), pick_indices]
                
                pick_correct = (pick_values > 0.) == pick_target
                total_pick_correct += torch.sum(pick_correct).float()
                total += b
                
                place_prediction = features['place']
                place_prediction = place_prediction.view(b, h*w)
                place_values, place_indices = torch.max(
                    place_prediction, dim=-1)
                place_target = place.view(b, h*w)[range(b), place_indices]
                
                place_correct = (place_values > 0.) == place_target
                total_place_correct += torch.sum(place_correct).float()
                
                total_both_correct += torch.sum(
                    pick_correct & place_correct).float()
                
                if i == 0:
                    for bb in range(b):
                        color_image = default_image_untransform(color[bb])
                        out_path = './vector_field_%04i_%04i.png'%(epoch, bb)
                        Image.fromarray(color_image).save(out_path)
                
            print('Test Pick Accuracy: %f'%(total_pick_correct/total))
            print('Test Place Accuracy: %f'%(total_place_correct/total))
            print('Test Both Accuracy: %f'%(total_both_correct/total))

if __name__ == '__main__':
    train(train_subset=None, test_subset=None)
