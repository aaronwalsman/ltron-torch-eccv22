#!/usr/bin/env python
import numpy

import torch
from torch.utils.data import Dataset, DataLoader

import tqdm

import PIL.Image as Image

import ltron.dataset.paths as paths
from ltron.visualization.drawing import draw_vector_field

from ltron.hierarchy import len_hierarchy
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
        self.paths = paths.get_dataset_paths(dataset, split, subset)
    
    def __len__(self):
        return len_hierarchy(self.paths)
    
    def __getitem__(self, index):
        color_path = self.paths['color'][index]
        color_image = Image.open(color_path)
        color = default_image_transform(color_image)
        
        pick_path = self.paths['pick'][index]
        pick = torch.BoolTensor(numpy.load(pick_path)).unsqueeze(0)
        
        place_path = self.paths['place'][index]
        place = torch.BoolTensor(numpy.load(place_path)).unsqueeze(0)
        
        offset_path = self.paths['offset'][index]
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
        'offset' : Conv2dStack(3, 256, 256, 2),
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
            
            num_pick = torch.sum(pick)
            if num_pick:
                offset_loss = torch.nn.functional.mse_loss(
                    features['offset'], offsets, reduction='none')
                offset_loss = torch.sum(offset_loss * pick) / num_pick
                offset_loss = torch.mean(offset_loss)
            else:
                offset_loss = 0
            
            iterate.set_description('pick: %.04f, offset: %.04f'%(
                float(pick_loss), float(offset_loss)))
            
            total_loss = pick_loss + offset_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        torch.save(model.state_dict(), './model_%04i.pt'%epoch)
        
        with torch.no_grad():
            iterate = tqdm.tqdm(test_loader)
            total_pick_correct = 0
            total_offset_error = 0
            total_place_correct = 0
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
                pick_values, pick_indices = torch.max(pick_prediction, dim=-1)
                pick_target = pick.view(b, h*w)[range(b), pick_indices]
                
                pick_correct = (pick_values > 0.) == pick_target
                total_pick_correct += torch.sum(pick_correct).float()
                total += b
                
                offset_prediction = features['offset']
                picked_offsets = offset_prediction.view(
                    b, 2, h*w)[range(b), :, pick_indices]
                scale = max(h,w)
                offset_target = offsets.view(b, 2, h*w)[
                    range(b), :, pick_indices]
                offset_error = picked_offsets - offset_target
                offset_error = torch.sum(offset_error**2, dim=1)**0.5 * scale
                total_offset_error += torch.sum(offset_error)
                
                start_y = torch.arange(h).unsqueeze(1).expand(h, w)
                start_x = torch.arange(w).unsqueeze(0).expand(h, w)
                start_yx = torch.stack((start_y, start_x), dim=0).unsqueeze(0)
                start_yx = start_yx.to(offset_prediction.device)
                dest_yx = start_yx + offset_prediction * scale
                dest_yx = torch.round(dest_yx).long()
                dest_yx[:,0] = torch.clamp(dest_yx[:,0], min=0, max=h-1)
                dest_yx[:,1] = torch.clamp(dest_yx[:,1], min=0, max=w-1)
                dest_yx = dest_yx.view(b, 2, h*w)[range(b),:,pick_indices]
                place = place.view(b, h, w)
                dest_place = place[range(b), dest_yx[:,0], dest_yx[:,1]]
                
                total_place_correct += torch.sum(dest_place).float()
                
                if i == 0:
                    for bb in range(b):
                        color_image = default_image_untransform(color[bb])
                        vector_field = (features['offset'][bb] * scale)
                        vector_field = vector_field.cpu().numpy()
                        vector_field = numpy.moveaxis(vector_field, 0, -1)
                        weight = torch.sigmoid(features['pick'][bb]).view(h,w)
                        weight = weight.cpu().numpy()
                        vector_color = (0,0,255)
                        draw_vector_field(
                            color_image,
                            vector_field,
                            weight,
                            vector_color,
                        )
                        color_path = './vector_field_%04i_%04i.png'%(epoch, bb)
                        Image.fromarray(color_image).save(color_path)
            
            print('Test Pick Accuracy: %f'%(total_pick_correct/total))
            print('Test Place Error: %f'%(total_offset_error/total))
            print('Test Place Correct: %f'%(total_place_correct/total))

if __name__ == '__main__':
    train(train_subset=None, test_subset=None)
