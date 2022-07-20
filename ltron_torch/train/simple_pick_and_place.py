#!/usr/bin/env python
import random
import os
import glob
import argparse

import numpy

import torch
from torch.utils.data import Dataset, DataLoader

import tqdm

from ltron_torch.models.positional_encoding import positional_encoding
from ltron_torch.models.transformer import (
    TrainConfig, TransformerConfig, TokenMapSequenceEncoder)

# Make Dataset =================================================================

dataset_path = './simple_pick_and_place'

def make_dataset(num_sequences=20000, bespoke=False):
    os.makedirs(dataset_path)
    for i in tqdm.tqdm(range(num_sequences)):
        target, brick_maps, actions = make_sequence()
        target_path = os.path.join(dataset_path, 'target_%06i.npy'%i)
        numpy.save(target_path, target)
        for j, (brick_map, action) in enumerate(zip(brick_maps, actions)):
            brick_map_path = os.path.join(
                dataset_path, 'observation_%06i_%04i.npy'%(i,j))
            numpy.save(brick_map_path, brick_map)

            action_path = os.path.join(
                dataset_path, 'action_%06i_%04i.npy'%(i,j))
            numpy.save(action_path, numpy.array(action))

def make_collision_blocks(positions):
    collisions = set()
    for y,x,o in positions:
        for yy in -1,0,1:
            for xx in -1,0,1:
                collisions.add((y+yy, x+xx))
    return collisions

def make_sequence():
    target_positions = random_positions()
    start_collisions = make_collision_blocks(target_positions)
    start_positions = random_positions(start_collisions)

    target_map = positions_to_brick_map(target_positions)
    sequence_maps = [positions_to_brick_map(start_positions)]
    
    actions = []
    for i in range(2):
        ty, tx, to = target_positions[i]
        sy, sx, so = start_positions[i]

        if ty != sy or tx != sx:
            actions.append((sy, sx, ty, tx, 0, 0, 1))
            sy = ty
            sx = tx
            start_positions[i] = (sy, sx, so)
            sequence_maps.append(positions_to_brick_map(start_positions))

        if to != so:
            actions.append((0, 0, 0, 0, sy, sx, 2))
            so = to
            start_positions[i] = (sy, sx, so)
            sequence_maps.append(positions_to_brick_map(start_positions))

    actions.append((0,0,0,0,0,0,0))

    return target_map, sequence_maps, actions

def random_positions(collisions=None):

    positions = []
    if collisions is None:
        collisions = set()
    for i in range(2):
        while True:
            o = random.randint(0,1)
            y = random.randint(0, 6)
            x = random.randint(0, 6)
            new_locations = position_to_map_locations((y,x,o))
            if not any([location in collisions for location in new_locations]):
                break

        collisions |= make_collision_blocks([(y,x,o)])
        positions.append((y,x,o))

    return positions

def position_to_map_locations(position):
    y,x,o = position
    locations = [(y,x)]
    if o == 0:
        locations.append((y+1,x))
    else:
        locations.append((y,x+1))

    return locations

def positions_to_brick_map(positions):
    brick_map = numpy.zeros((8,8), dtype=numpy.long)

    for i, position in enumerate(positions, start=1):
        map_locations = position_to_map_locations(position)
        for y,x in map_locations:
            brick_map[y,x] = i
    
    return brick_map


# Dataset Loader ===============================================================

def extract_path_indices(path, num_indices):
    parts = path.split('_')[-num_indices:]
    parts = tuple([int(part.split('.')[0]) for part in parts])
    return parts

class SimpleDataset(Dataset):
    def __init__(self, train=True):
        observation_paths = glob.glob(
            os.path.join(dataset_path, 'observation_*.npy'))
        if train:
            self.observation_paths = [
                path for path in observation_paths
                if extract_path_indices(path, 2)[0] < 15000]
        else:
            self.observation_paths = [
                path for path in observation_paths
                if extract_path_indices(path, 2)[0] >= 15000]
    
    def __getitem__(self, index):
        observation_path = self.observation_paths[index]
        target_path = observation_path.replace('observation_', 'target_')
        target_path = target_path.rsplit('_', 1)[0] + '.npy'
        action_path = observation_path.replace('observation_', 'action_')
        
        observation = torch.LongTensor(numpy.load(observation_path))
        target = torch.LongTensor(numpy.load(target_path))
        action = torch.LongTensor(numpy.load(action_path))
        
        return observation, target, action
    
    def __len__(self):
        return len(self.observation_paths)


# Model Utils ==================================================================
def get_num_model_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


# Dense Pick ===================================================================

class BespokeDensePickModel(torch.nn.Module):
    def __init__(
        self,
        channels=256,
    ):
        super(BespokeDensePickModel, self).__init__()
        self.register_buffer(
            'positional_encoding',
            positional_encoding(channels, 5000).unsqueeze(1)
        )
        
        self.in_embedding = torch.nn.Embedding(3, channels)
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            4,
            channels,
            0.5,
        )
        self.transformer = torch.nn.TransformerEncoder(
            transformer_layer, 6)
        
        self.pick_linear = torch.nn.Linear(channels, 3)
    
    def forward(self, x):
        s, hw, b = x.shape
        x = x.view(s*hw, b)
        x = self.in_embedding(x)
        x = x + self.positional_encoding[:hw]
        
        x = self.transformer(x)
        
        pick = self.pick_linear(x)
        
        return pick

def train_dense_pick(bespoke=False):
    print('training dense pick')
    print('making dataset')
    train_dataset = SimpleDataset(train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
    )
    
    if bespoke:
        print('building bespoke model')
        model = BespokeDensePickModel().cuda()
        optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    else:
        print('building general model')
        config = TransformerConfig(
            vocabulary=3,
            map_height=8,
            map_widthw=8,
            decoder_tokens=0,
            decode_input=True,
            
            num_blocks=6,
            channels = 256,
            residual_channels = 256,
            num_heads = 4,
            decoder_channels = 3,
            
            learned_positional_encoding = True,
            init_weights = True,
        )
        model = TokenMapSequenceEncoder(config).cuda()
        train_config = TrainConfig()
        optimizer = model.configure_optimizer(train_config)
    print('parameters: %i'%get_num_model_parameters(model))
    
    running_pick_accuracy = 0.0
    for epoch in range(1, 6):
        print('epoch: %i'%epoch)
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for x, q, y in iterate:
            x = x.cuda()
            b, h, w = x.shape
            
            total_loss = 0.
            
            pick = model(x.view(b, h*w).permute(1,0).unsqueeze(0))
            pick = pick.permute(1,2,0)
            b = pick.shape[0]
            pick = pick.reshape(b,3,8,8)
            
            pick_target = torch.zeros((b,8,8), dtype=torch.long)
            one_spots = torch.nonzero(x == 1, as_tuple=False)[::2][:,1:]
            pick_target[torch.arange(b), one_spots[:,0], one_spots[:,1]] = 1
            two_spots = torch.nonzero(x == 2, as_tuple=False)[::2][:,1:]
            pick_target[torch.arange(b), two_spots[:,0], two_spots[:,1]] = 2
            pick_target = pick_target.cuda()
            
            pick_loss = torch.nn.functional.cross_entropy(pick, pick_target)
            total_loss = total_loss + pick_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pick_max = torch.argmax(pick, dim=1)
            pick_correct = pick_max == pick_target
            b = pick_correct.shape[0]
            pick_accuracy = torch.sum(pick_correct).float() / (b * 8 * 8)
            running_pick_accuracy = (
                running_pick_accuracy * 0.9 + float(pick_accuracy) * 0.1)
            
            iterate.set_description('m: %.04f'%(running_pick_accuracy))


# Sparse Pick ==================================================================

class BespokeSparsePickModel(torch.nn.Module):
    def __init__(
        self,
        channels=256,
    ):
        super(BespokeSparsePickModel, self).__init__()
        self.register_buffer(
            'sequence_encoding',
            positional_encoding(channels, 5000).unsqueeze(1)
        )
        
        self.x_embedding = torch.nn.Embedding(3, channels)
        self.d_embedding = torch.nn.Embedding(2, channels)
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            4,
            channels,
            0.5,
        )
        self.transformer = torch.nn.TransformerEncoder(
            transformer_layer, 6)
        
        self.pick_linear = torch.nn.Linear(channels, 8+8)
    
    def forward(self, x):
        s, hw, b = x.shape
        x = x.view(s*hw, b)
        x = self.x_embedding(x)
        c = x.shape[-1]
        d = self.d_embedding(torch.arange(2).cuda()).unsqueeze(1).expand(2,b,c)
        dx = torch.cat((d,x), dim=0)
        dx = dx + self.sequence_encoding[:2+hw]
        
        dx = self.transformer(dx)
        
        pick = self.pick_linear(dx[:2])
        
        return pick

def train_sparse_pick(bespoke=False):
    print('training sparse pick')
    print('making dataset')
    train_dataset = SimpleDataset(train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
    )
    
    if bespoke:
        print('building bespoke model')
        model = BespokeSparsePickModel().cuda()
        optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    else:
        print('building general model')
        config = TransformerConfig(
            vocabulary=3,
            map_height=8,
            map_width=8,
            decoder_tokens=2,
            decode_input=False,
            
            channels=256,
            residual_channels=256,
            num_blocks=6,
            num_heads=4,
            decoder_channels=8+8,
            
            learned_positional_encoding = True,
            
            embedding_dropout = 0.0,
            attention_dropout = 0.5,
            residual_dropout = 0.5,
        )
        model = TokenMapSequenceEncoder(config).cuda()
        train_config = TrainConfig()
        optimizer = model.configure_optimizer(train_config)
    print('parameters: %i'%get_num_model_parameters(model))
    
    running_pick_accuracy = 0.0
    for epoch in range(1, 6):
        print('epoch: %i'%epoch)
        print('train')
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for x, q, y in iterate:
            x = x.cuda()
            b, h, w = x.shape
            
            total_loss = 0.
            
            pick = model(x.view(1, b, h*w).permute(0,2,1))
            b = pick.shape[1]
            pick_one = pick[0].view(b, 8, 2)
            pick_two = pick[1].view(b, 8, 2)
            pick = torch.cat((pick_one, pick_two), dim=0)
            
            one_spots = torch.nonzero(x == 1, as_tuple=False)[::2][:,1:]
            two_spots = torch.nonzero(x == 2, as_tuple=False)[::2][:,1:]
            pick_target = torch.cat((one_spots, two_spots), dim=0)
            
            pick_loss = torch.nn.functional.cross_entropy(pick, pick_target)
            total_loss = total_loss + pick_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pick_max = torch.argmax(pick, dim=1)
            pick_correct = pick_max == pick_target
            b = pick_correct.shape[0]
            pick_accuracy = torch.sum(pick_correct).float() / (b * 2)
            running_pick_accuracy = (
                running_pick_accuracy * 0.9 + float(pick_accuracy) * 0.1)
            
            iterate.set_description('m: %.04f'%(running_pick_accuracy))

# AB Match =====================================================================

class BespokeABMatchModel(torch.nn.Module):
    def __init__(
        self,
        channels=256,
    ):
        super(BespokeABMatchModel, self).__init__()
        self.register_buffer(
            'sequence_encoding',
            positional_encoding(channels, 5000).unsqueeze(1)
        )
        
        self.in_embedding = torch.nn.Embedding(3, channels)
        self.xq_embedding = torch.nn.Embedding(2, channels)
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            4,
            channels,
            0.5,
        )
        self.transformer = torch.nn.TransformerEncoder(
            transformer_layer, 6)
        
        self.match_linear = torch.nn.Linear(channels, 2)
    
    def forward(self, x, q):
        b, h, w = x.shape
        
        x = x.view(b, h*w).permute(1,0)
        x = self.in_embedding(x)
        x = x + self.sequence_encoding[:h*w]
        x = x + self.xq_embedding.weight[0].unsqueeze(0).unsqueeze(1)
        
        q = q.view(b, h*w).permute(1,0)
        q = self.in_embedding(q)
        q = q + self.sequence_encoding[:h*w]
        q = q + self.xq_embedding.weight[1].unsqueeze(0).unsqueeze(1)
        
        # sponge mask
        #mask = torch.ones((h*w*2, h*w*2), dtype=torch.bool)
        #mask[:h*w,:h*w] = False
        #mask[h*w:,h*w:] = False
        #mask[torch.arange(h*w, h*w*2), torch.arange(h*w)] = False
        #mask[torch.arange(h*w), torch.arange(h*w,h*w*2)] = False
        #mask = mask.cuda()
        mask = None
        
        qx = torch.cat((q,x), dim=0)
        qx = self.transformer(qx, mask=mask)
        
        match = self.match_linear(qx[:64])
        
        return match

def train_ab_match():
    print('training ab match')
    print('making dataset')
    train_dataset = SimpleDataset(train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
    )
    
    print('building model')
    model = BespokeABMatchModel().cuda()
    print('parameters: %i'%get_num_model_parameters(model))
    
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    
    running_match_accuracy = 0.0
    for epoch in range(1, 6):
        print('epoch: %i'%epoch)
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for x, q, y in iterate:
            x = x.cuda()
            q = q.cuda()
            
            total_loss = 0.
            
            match = model(x, q)
            match = match.permute(1,2,0)
            b = match.shape[0]
            match = match.reshape(b,2,8,8)
            match_target = (x != q).long()
            
            match_loss = torch.nn.functional.cross_entropy(match, match_target)
            total_loss = total_loss + match_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            match_max = torch.argmax(match, dim=1)
            match_correct = match_max == match_target
            b = match_correct.shape[0]
            match_accuracy = torch.sum(match_correct).float() / (b * 8 * 8)
            running_match_accuracy = (
                running_match_accuracy * 0.9 + float(match_accuracy) * 0.1)
            
            iterate.set_description('m: %.04f'%(running_match_accuracy))


# Sparse Pick and Place ========================================================

class BespokeSparsePickAndPlaceModel(torch.nn.Module):
    def __init__(
        self,
        channels=256,
    ):
        super(BespokeSparsePickAndPlaceModel, self).__init__()
        self.register_buffer(
            'sequence_encoding',
            positional_encoding(channels, 5000).unsqueeze(1)
        )
        
        self.qx_embedding = torch.nn.Embedding(3, channels)
        self.id_embedding = torch.nn.Embedding(2, channels)
        self.d_embedding = torch.nn.Embedding(2, channels)
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            4,
            channels,
            0.5,
        )
        self.transformer = torch.nn.TransformerEncoder(
            transformer_layer, 6)
        
        self.pick_and_place_linear = torch.nn.Linear(channels, 64+64)
    
    def forward(self, x, q):
        b, h, w = x.shape
        
        x = x.view(b, h*w).permute(1,0)
        x = self.qx_embedding(x)
        x = x + self.id_embedding.weight[0].unsqueeze(0).unsqueeze(1)
        
        q = q.view(b, h*w).permute(1,0)
        q = self.qx_embedding(q)
        q = q + self.id_embedding.weight[1].unsqueeze(0).unsqueeze(1)
        
        c = x.shape[-1]
        d = self.d_embedding(torch.arange(2).cuda()).unsqueeze(1).expand(2,b,c)
        dqx = torch.cat((d,q,x), dim=0)
        dqx = dqx + self.sequence_encoding[:2+h*w+h*w]
        
        mask = None
        
        dqx = self.transformer(dqx, mask=mask)
        
        pick_and_place = self.pick_and_place_linear(dqx[:2])
        
        return pick_and_place

def train_sparse_pick_and_place():
    print('training sparse pick and place')
    print('making dataset')
    train_dataset = SimpleDataset(train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
    )
    
    print('building model')
    model = BespokeSparsePickAndPlaceModel().cuda()
    print('parameters: %i'%get_num_model_parameters(model))
    
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    
    running_accuracy = 0.0
    for epoch in range(1, 6):
        print('epoch: %i'%epoch)
        print('train')
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for x, q, y in iterate:
            x = x.cuda()
            q = q.cuda()
            
            total_loss = 0.
            
            pick_and_place = model(x,q)
            b = pick_and_place.shape[1]
            pick_and_place_one = pick_and_place[0].view(b, 64, 2)
            pick_and_place_two = pick_and_place[1].view(b, 64, 2)
            pick_and_place = torch.cat(
                (pick_and_place_one, pick_and_place_two), dim=0)
            
            pick_one_spots = torch.nonzero(x == 1, as_tuple=False)[::2][:,1:]
            pick_one_spots = pick_one_spots[:,0] * 8 + pick_one_spots[:,1]
            pick_two_spots = torch.nonzero(x == 2, as_tuple=False)[::2][:,1:]
            pick_two_spots = pick_two_spots[:,0] * 8 + pick_two_spots[:,1]
            pick_target = torch.cat((pick_one_spots, pick_two_spots), dim=0)
            
            place_one_spots = torch.nonzero(q == 1, as_tuple=False)[::2][:,1:]
            place_one_spots = place_one_spots[:,0] * 8 + place_one_spots[:,1]
            place_two_spots = torch.nonzero(q == 2, as_tuple=False)[::2][:,1:]
            place_two_spots = place_two_spots[:,0] * 8 + place_two_spots[:,1]
            place_target = torch.cat((place_one_spots, place_two_spots), dim=0)
            
            pick_and_place_target = torch.stack(
                (pick_target, place_target), dim=-1)
            
            pick_and_place_loss = torch.nn.functional.cross_entropy(
                pick_and_place, pick_and_place_target)
            total_loss = total_loss + pick_and_place_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pick_and_place_max = torch.argmax(pick_and_place, dim=1)
            pick_and_place_correct = pick_and_place_max == pick_and_place_target
            b = pick_and_place_correct.shape[0]
            pick_and_place_accuracy = torch.sum(
                pick_and_place_correct).float() / (b * 2)
            running_accuracy = (
                running_accuracy * 0.9 + float(pick_and_place_accuracy) * 0.1)
            
            iterate.set_description('m: %.04f'%(running_accuracy))

# Super Sparse Pick Place Rotate ===============================================

'''
This method doesn't work.  In general the approach seems untennable:
The model has to recognize globally which piece is the one to move next, then also identify if it needs to be translated or rotated or do-nothing-ed.  I don't like full dense prediction, but I can also see why  super-sparse prediction is going to be bad.  We need a small, but not too-small set of tokens that can just do local reasoning without having to worry about the entire scene.  We've seen that single tokens can reason about matching across spatial locations when they only have to pay attention to one thing.

This begs the question: how do we assign labels to these amorphous sparse tokens?  We can do hungarian matching ala DETR or... something else we figure out.  Right now we have the benefit of only ever having two distinct bricks, so assigning a token to each is fine.
'''
class BespokeSuperSparsePickPlaceRotateModel(torch.nn.Module):
    def __init__(
        self,
        channels=256,
    ):
        super(BespokeSuperSparsePickPlaceRotateModel, self).__init__()
        self.register_buffer(
            'sequence_encoding',
            positional_encoding(channels, 5000).unsqueeze(1)
        )
        
        self.qx_embedding = torch.nn.Embedding(3, channels)
        self.id_embedding = torch.nn.Embedding(2, channels)
        self.d_embedding = torch.nn.Embedding(2, channels)
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            4,
            channels,
            0.5,
        )
        self.transformer = torch.nn.TransformerEncoder(
            transformer_layer, 6)
        
        self.pick_place_rotate_linear = torch.nn.Linear(channels, 64+64+64)
    
    def forward(self, x, q):
        b, h, w = x.shape
        
        x = x.view(b, h*w).permute(1,0)
        x = self.qx_embedding(x)
        x = x + self.id_embedding.weight[0].unsqueeze(0).unsqueeze(1)
        
        q = q.view(b, h*w).permute(1,0)
        q = self.qx_embedding(q)
        q = q + self.id_embedding.weight[1].unsqueeze(0).unsqueeze(1)
        
        c = x.shape[-1]
        d = self.d_embedding(torch.arange(1).cuda()).unsqueeze(1).expand(1,b,c)
        dqx = torch.cat((d,q,x), dim=0)
        dqx = dqx + self.sequence_encoding[:1+h*w+h*w]
        
        mask = None
        
        dqx = self.transformer(dqx, mask=mask)
        
        pick_place_rotate = self.pick_place_rotate_linear(dqx[0])
        
        return pick_place_rotate

def train_super_sparse_pick_place_rotate():
    print('training super sparse pick, place and rotate')
    print('making dataset')
    train_dataset = SimpleDataset(train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
    )
    
    print('building model')
    model = BespokeSuperSparsePickPlaceRotateModel().cuda()
    print('parameters: %i'%get_num_model_parameters(model))
    
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    
    running_accuracy = 0.0
    for epoch in range(1, 6):
        print('epoch: %i'%epoch)
        print('train')
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for x, q, y in iterate:
            x = x.cuda()
            q = q.cuda()
            y = y.cuda()
            
            total_loss = 0.
            
            pick_place_rotate = model(x,q)
            b = pick_place_rotate.shape[0]
            pick_place_rotate = pick_place_rotate.view(b, 64, 3)
            
            target = y[:,:6]
            target = target[:,::2] * 8 + target[:,1::2]
            
            pick_place_rotate_loss = torch.nn.functional.cross_entropy(
                pick_place_rotate, target)
            total_loss = total_loss + pick_place_rotate_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pick_place_rotate_max = torch.argmax(pick_place_rotate, dim=1)
            correct = pick_place_rotate_max == target
            b = correct.shape[0]
            accuracy = torch.sum(correct).float() / (b * 3)
            running_accuracy = (
                running_accuracy * 0.9 + float(accuracy) * 0.1)
            
            iterate.set_description('m: %.04f'%(running_accuracy))


# Sparse Pick Place Rotate =====================================================

class BespokeSparsePickPlaceRotateModel(torch.nn.Module):
    def __init__(
        self,
        channels=256,
    ):
        super(BespokeSparsePickPlaceRotateModel, self).__init__()
        self.register_buffer(
            'sequence_encoding',
            positional_encoding(channels, 5000).unsqueeze(1)
        )
        
        self.qx_embedding = torch.nn.Embedding(3, channels)
        self.id_embedding = torch.nn.Embedding(2, channels)
        self.d_embedding = torch.nn.Embedding(2, channels)
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            4,
            channels,
            0.5,
        )
        self.transformer = torch.nn.TransformerEncoder(
            transformer_layer, 6)
        
        self.pick_place_rotate_linear = torch.nn.Linear(channels, 64+64+4)
    
    def forward(self, x, q):
        b, h, w = x.shape
        
        x = x.view(b, h*w).permute(1,0)
        x = self.qx_embedding(x)
        x = x + self.id_embedding.weight[0].unsqueeze(0).unsqueeze(1)
        
        q = q.view(b, h*w).permute(1,0)
        q = self.qx_embedding(q)
        q = q + self.id_embedding.weight[1].unsqueeze(0).unsqueeze(1)
        
        c = x.shape[-1]
        d = self.d_embedding(torch.arange(2).cuda()).unsqueeze(1).expand(2,b,c)
        dqx = torch.cat((d,q,x), dim=0)
        dqx = dqx + self.sequence_encoding[:2+h*w+h*w]
        
        mask = None
        
        dqx = self.transformer(dqx, mask=mask)
        
        pick_place_rotate = self.pick_place_rotate_linear(dqx[:2])
        
        return pick_place_rotate

def train_sparse_pick_place_rotate():
    print('training sparse pick, place and rotate')
    print('making dataset')
    train_dataset = SimpleDataset(train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
    )
    
    print('building model')
    model = BespokeSparsePickPlaceRotateModel().cuda()
    print('parameters: %i'%get_num_model_parameters(model))
    
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    
    running_location_accuracy = 0.0
    running_mode_accuracy = 0.0
    for epoch in range(1, 6):
        print('epoch: %i'%epoch)
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for x, q, y in iterate:
            x = x.cuda()
            q = q.cuda()
            
            total_loss = 0.
            
            pick_place_rotate = model(x,q)
            
            b = pick_place_rotate.shape[1]
            one_mode = pick_place_rotate[0][:,:4]
            one_locations = pick_place_rotate[0][:,4:].view(b, 64, 2)
            two_mode = pick_place_rotate[1][:,:4]
            two_locations = pick_place_rotate[1][:,4:].view(b, 64, 2)
            
            one_two_mode = torch.cat((one_mode, two_mode), dim=0).view(b*2,2,2)
            one_two_locations = torch.cat((one_locations, two_locations), dim=0)
            
            x_one_spots = torch.nonzero(x == 1, as_tuple=False)
            pick_one_spots = x_one_spots[::2][:,1:]
            pick_one_spots = pick_one_spots[:,0] * 8 + pick_one_spots[:,1]
            x_two_spots = torch.nonzero(x == 2, as_tuple=False)
            pick_two_spots = x_two_spots[::2][:,1:]
            pick_two_spots = pick_two_spots[:,0] * 8 + pick_two_spots[:,1]
            pick_target = torch.cat((pick_one_spots, pick_two_spots), dim=0)
            
            pick_one_tail_spots = x_one_spots[1::2][:,1:]
            pick_one_tail_spots = (
                pick_one_tail_spots[:,0] * 8 + pick_one_tail_spots[:,1])
            pick_two_tail_spots = x_two_spots[1::2][:,1:]
            pick_two_tail_spots = (
                pick_two_tail_spots[:,0] * 8 + pick_two_tail_spots[:,1])
            pick_tail = torch.cat(
                (pick_one_tail_spots, pick_two_tail_spots), dim=0)
            
            q_one_spots = torch.nonzero(q == 1, as_tuple=False)
            place_one_spots = q_one_spots[::2][:,1:]
            place_one_spots = place_one_spots[:,0] * 8 + place_one_spots[:,1]
            q_two_spots = torch.nonzero(q == 2, as_tuple=False)
            place_two_spots = q_two_spots[::2][:,1:]
            place_two_spots = place_two_spots[:,0] * 8 + place_two_spots[:,1]
            place_target = torch.cat((place_one_spots, place_two_spots), dim=0)
            
            place_one_tail_spots = q_one_spots[1::2][:,1:]
            place_one_tail_spots = (
                place_one_tail_spots[:,0] * 8 + place_one_tail_spots[:,1])
            place_two_tail_spots = q_two_spots[1::2][:,1:]
            place_two_tail_spots = (
                place_two_tail_spots[:,0] * 8 + place_two_tail_spots[:,1])
            place_tail = torch.cat(
                (place_one_tail_spots, place_two_tail_spots), dim=0)
            
            pick_place_target = torch.stack(
                (pick_target, place_target), dim=-1)
            
            pick_place_loss = torch.nn.functional.cross_entropy(
                one_two_locations, pick_place_target)
            total_loss = total_loss + pick_place_loss
            
            requires_translate = (pick_target != place_target)
            requires_rotate = (pick_tail != place_tail)
            '''
            mode_target = (
                requires_translate * 1 +
                ((~requires_translate) & requires_rotate) * 2
            )
            '''
            mode_target = torch.stack(
                (requires_translate, requires_rotate), dim=-1).long()
            mode_loss = torch.nn.functional.cross_entropy(
                one_two_mode, mode_target)
            total_loss = total_loss + mode_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pick_place_max = torch.argmax(one_two_locations, dim=1)
            pick_place_correct = pick_place_max == pick_place_target
            b = pick_place_correct.shape[0]
            pick_place_accuracy = torch.sum(
                pick_place_correct).float() / (b * 2)
            running_location_accuracy = (
                running_location_accuracy * 0.9 +
                float(pick_place_accuracy) * 0.1
            )
            
            mode_max = torch.argmax(one_two_mode, dim=1)
            mode_correct = mode_max == mode_target
            mode_accuracy = torch.sum(mode_correct).float() / (b * 2)
            running_mode_accuracy = (
                running_mode_accuracy * 0.9 +
                float(mode_accuracy) * 0.1
            )
            
            iterate.set_description('l:%.04f m:%0.04f'%(
                running_location_accuracy, running_mode_accuracy)
            )

# Run ==========================================================================

parser = argparse.ArgumentParser()

parser.add_argument('experiment', type=str)
parser.add_argument('--bespoke', action='store_true')

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    experiment_functions = {
        'make_dataset' : make_dataset,
        'dense_pick' : train_dense_pick,
        'sparse_pick' : train_sparse_pick,
        'ab_match' : train_ab_match,
        'sparse_pick_and_place' : train_sparse_pick_and_place,
        'sparse_pick_place_rotate' : train_sparse_pick_place_rotate,
    }
    
    assert args.experiment in experiment_functions
    
    experiment_functions[args.experiment](bespoke=args.bespoke)
