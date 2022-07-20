#!/usr/bin/env python
import numpy

import torch
from torch.utils.data import Dataset, DataLoader

import tqdm

from skimage.draw import line

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
        
        pos_path = color_path.replace(
            'color_', 'pos_snaps_').replace('.png', '.npy')
        pos_snaps = numpy.load(pos_path)
        pos_snaps = numpy.moveaxis(pos_snaps, -1, 0)
        pos_snaps = torch.LongTensor(pos_snaps)
        
        neg_path = pos_path.replace('pos_snaps_', 'neg_snaps_')
        neg_snaps = numpy.load(neg_path)
        neg_snaps = numpy.moveaxis(neg_snaps, -1, 0)
        neg_snaps = torch.LongTensor(neg_snaps)
        
        return color, pos_snaps, neg_snaps

class MatchModel(torch.nn.Module):
    def __init__(self):
        super(MatchModel, self).__init__()
        fcn = SimpleFCN(
            pretrained=True,
            compute_single=False,
        )
        dense_heads = {
            'pick' : Conv2dStack(3, 256, 256, 1),
            'place' : Conv2dStack(3, 256, 256, 1),
            'x' : torch.nn.Identity(),
        }
        self.step_model = StepModel(
            fcn, dense_heads=dense_heads, single_heads={})
        
        self.matching_head = Conv2dStack(3, 512, 256, 1)
    
    def forward(self, x, topk=64):
        features = self.step_model(x)
        
        b, c, h, w = features['x'].shape
        
        pick = features['pick'].view(b, h*w)
        pick_values, pick_indices = torch.topk(pick, topk, dim=-1)
        pick_indices_expand = pick_indices.unsqueeze(1).expand(b, c, topk)
        
        pick_features = features['x'].view(b, c, h*w)
        pick_features = torch.gather(pick_features, -1, pick_indices_expand)
        pick_features = pick_features.unsqueeze(-1).expand(b, c, topk, topk)
        
        place = features['place'].view(b, h*w)
        place_values, place_indices = torch.topk(place, topk, dim=-1)
        place_indices_expand = place_indices.unsqueeze(1).expand(b, c, topk)
        
        place_features = features['x'].view(b, c, h*w)
        place_features = torch.gather(place_features, -1, place_indices_expand)
        place_features = place_features.unsqueeze(-2).expand(b, c, topk, topk)
        
        pick_and_place = torch.cat((pick_features, place_features), dim=1)
        #pick_and_place = (pick_features - place_features)**2
        
        match = self.matching_head(pick_and_place)
        
        features['match'] = match
        features['pick_indices'] = pick_indices
        features['place_indices'] = place_indices
        
        return features

def train(
    epochs = 10,
    batch_size=16,
    train_subset=None,
    test_subset=None,
    topk = 64,
):
    
    train_dataset = FrameDataset(
        'snap_four_frames',
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
        'snap_four_frames',
        'test',
        subset=test_subset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    
    model = MatchModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    pick_place_pairs = (
        ((2, 0), (1, 8)),
        ((3, 0), (1, 7)),
        ((4, 0), (1, 6)),
        ((5, 0), (1, 5)),
    )
    
    for epoch in range(1, epochs+1):
        iterate = tqdm.tqdm(train_loader)
        for color, pos_snaps, neg_snaps in iterate:
            color = color.cuda()
            pos_snaps = pos_snaps.cuda()
            neg_snaps = neg_snaps.cuda()
            
            features = model(color, topk=topk)
            
            b, c, h, w = pos_snaps.shape
            pick_map = torch.zeros(b, h, w, dtype=torch.bool).cuda()
            place_map = torch.zeros(b, h, w, dtype=torch.bool).cuda()
            
            match_map = torch.zeros(b, topk, topk, dtype=torch.bool).cuda()
            pick_topk = torch.gather(
                neg_snaps.view(b, 2, h*w),
                -1,
                features['pick_indices'].unsqueeze(1).expand(b,2,topk))
            place_topk = torch.gather(
                pos_snaps.view(b, 2, h*w),
                -1,
                features['place_indices'].unsqueeze(1).expand(b,2,topk))
            
            for pick, place in pick_place_pairs:
                pick_instance, pick_snap = pick
                place_instance, place_snap = place
                pick_map |= (
                    (neg_snaps[:,0] == pick_instance) &
                    (neg_snaps[:,1] == pick_snap)
                )
                place_map |= (
                    (pos_snaps[:,0] == place_instance) &
                    (pos_snaps[:,1] == place_snap)
                )
                
                pick_match = (
                    pick_topk == torch.LongTensor(pick).unsqueeze(-1).cuda())
                pick_match = pick_match[:,0] & pick_match[:,1]
                
                place_match = (
                    place_topk == torch.LongTensor(place).unsqueeze(-1).cuda())
                place_match = place_match[:,0] & place_match[:,1]
                
                match_map |= (
                    pick_match.unsqueeze(-1) & place_match.unsqueeze(-2))
            
            pick_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                features['pick'], pick_map.unsqueeze(1).float())
            
            place_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                features['place'], place_map.unsqueeze(1).float())
            
            match_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                features['match'], match_map.unsqueeze(1).float())
            
            iterate.set_description('pk: %.04f, pl: %.04f, m: %.04f'%(
                float(pick_loss), float(place_loss), float(match_loss)))
            
            total_loss = pick_loss + place_loss + match_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        torch.save(model.state_dict(), './model_%04i.pt'%epoch)
        
        with torch.no_grad():
            iterate = tqdm.tqdm(test_loader)
            total_pick_correct = 0
            total_place_correct = 0
            total_match_correct = 0
            total = 0
            for i, (color, pos_snaps, neg_shaps) in enumerate(iterate):
                color = color.cuda()
                pos_snaps = pos_snaps.cuda()
                neg_snaps = neg_snaps.cuda()
                
                features = model(color)
                
                if i == 0:
                    pick_prediction = torch.sigmoid(features['pick'])
                    y_scale = color.shape[2] // pick_prediction.shape[2]
                    x_scale = color.shape[3] // pick_prediction.shape[3]
                    upsampled_pick_prediction = torch.repeat_interleave(
                        pick_prediction, y_scale, dim=2)
                    upsampled_pick_prediction = torch.repeat_interleave(
                        upsampled_pick_prediction, x_scale, dim=3)
                    
                    place_prediction = torch.sigmoid(features['place'])
                    upsampled_place_prediction = torch.repeat_interleave(
                        place_prediction, y_scale, dim=2)
                    upsampled_place_prediction = torch.repeat_interleave(
                        upsampled_place_prediction, x_scale, dim=3)
                    
                    match_prediction = torch.sigmoid(features['match'])
                    
                    for j in range(b):
                        color_image = default_image_untransform(color[j])
                        '''
                        blue = upsampled_pick_prediction[j,0].unsqueeze(-1)
                        blue = blue.cpu().numpy()
                        color_image = (
                            color_image * (1. - blue) + (0,0,255) * blue)
                        green = upsampled_place_prediction[j,0].unsqueeze(-1)
                        green = green.cpu().numpy()
                        color_image = (
                            color_image * (1. - green) + (0,255,0) * green)
                        color_image = color_image.astype(numpy.uint8)
                        '''
                        
                        b, c, h, w = features['pick'].shape
                        #vector_field = numpy.zeros((h, w, 2))
                        #vector_weight = numpy.zeros((h,w))
                        
                        match = match_prediction[j,0].cpu().numpy()
                        pick_indices = features['pick_indices'].cpu().numpy()
                        pick_y = pick_indices[j] // w
                        pick_x = pick_indices[j] % w
                        place_indices = features['place_indices'].cpu().numpy()
                        place_y = place_indices[j] // w
                        place_x = place_indices[j] % w
                        for pick_id in range(topk):
                            for place_id in range(topk):
                                y, x = line(
                                    int(pick_y[pick_id]*y_scale + y_scale/2),
                                    int(pick_x[pick_id]*x_scale + x_scale/2),
                                    int(place_y[place_id]*y_scale + y_scale/2),
                                    int(place_x[place_id]*x_scale + x_scale/2),
                                )
                                weight = match[pick_id, place_id]
                                color_image[y,x] = (
                                    color_image[y,x] * (1. - weight) +
                                    numpy.array((255,0,0)) * weight)
                        #vector_field[source_y, source_x, 0] = offset_y
                        #vector_field[source_y, source_x, 1] = offset_x
                        #draw_vector_field(
                        #    color_image,
                        #    vector_field,
                        #    vector_weight,
                        #    (255, 0, 0)
                        #)
                        
                        Image.fromarray(
                            color_image).save('./pp_%04i_%04i.png'%(epoch, j))
                        
            #print('Test Pick Accuracy: %f'%(total_pick_correct/total))
            #print('Test Place Accuracy: %f'%(total_place_correct/total))
            #print('Test Both Accuracy: %f'%(total_both_correct/total))

if __name__ == '__main__':
    train(train_subset=None, test_subset=None)
