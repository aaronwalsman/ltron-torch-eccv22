import random
import os

import numpy

import torch
from torch.nn.functional import cross_entropy

import tqdm

from ltron.compression import batch_deduplicate_from_masks

from ltron_torch.gym_tensor import (
    default_tile_transform,
    default_image_transform,
)
from ltron_torch.models.padding import cat_padded_seqs
from ltron_torch.interface.break_and_make import (
    BreakAndMakeInterfaceConfig,
    BreakAndMakeInterface,
)

class BreakAndMakeStubnetTransformerInterfaceConfig(
    BreakAndMakeInterfaceConfig
):
    misclick_augmentation = 0.15
    tile_shift_augmentation = 2
    tile_h = 16
    tile_w = 16
    table_h = 256
    table_w = 256
    hand_h = 96
    hand_w = 96
    
    #table_decode_h = 64
    #table_decode_w = 64
    #hand_decode_h = 24
    #hand_decode_w = 24
    
    #def set_dependents(self):
    #    assert self.table_h % self.tile_h == 0
    #    assert self.table_w % self.table_w == 0
    #    self.table_tiles_h = self.table_h // self.tile_h
    #    self.table_tiles_w = self.table_w // self.tile_w
    #    
    #    assert self.hand_h % self.tile_h == 0
    #    assert self.hand_w % self.tile_w == 0
    #    self.hand_tiles_h = self.hand_h // self.tile_h
    #    self.hand_tiles_w = self.hand_w // self.tile_w

class BreakAndMakeStubnetTransformerInterface(BreakAndMakeInterface):
    
    def observation_to_tensors(self, observation, action, pad):
        
        # initialize x
        x = {}
        
        # get the device
        device = next(self.model.parameters()).device
        
        # shape
        s, b, h, w, c = observation['table_color_render'].shape
        
        # get the insert activate frames
        if action is not None:
            insert_activate = (action['insert_brick']['shape'] != 0).reshape(-1)
            insert_activate = insert_activate.astype(numpy.bool)
        else:
            insert_activate = numpy.ones(s*b, dtype=numpy.bool)
        x['insert_activate'] = torch.BoolTensor(insert_activate).to(device)
        
        # process table images
        table_color_render = observation['table_color_render'].reshape(
            s*b, h, w, c)
        if action is not None:
            table_activate = action['table_cursor']['activate'].reshape(-1)
            table_activate = table_activate.astype(numpy.bool)
        else:
            table_activate = numpy.ones(s*b, dtype=numpy.bool)
        #table_image = torch.stack([
        #    default_image_transform(image) for image in table_color_render
        #]).reshape(s*b, c, h, w).to(device)
        x['table_image'] = torch.stack([
            default_image_transform(image)
            for a, image in zip(table_activate, table_color_render)
            if a
        ]).to(device)
        x['table_cursor_activate'] = torch.BoolTensor(
            table_activate).view(s, b).to(device)
        
        #table_activate = torch.LongTensor(
        #    action['table_cursor']['activate']).to(device)
        #x['table_image'] = table_image[table_activate.view(-1)]
        
        # process hand images
        s, b, h, w, c = observation['hand_color_render'].shape
        hand_color_render = observation['hand_color_render'].reshape(
            s*b, h, w, c)
        if action is not None:
            hand_activate = action['hand_cursor']['activate'].reshape(-1)
            hand_activate = hand_activate.astype(numpy.bool)
        else:
            hand_activate = numpy.ones(s*b, dtype=numpy.bool)
        #hand_image = torch.stack([
        #    default_image_transform(image) for image in hand_color_render
        #]).reshape(s, b, c, h, w).to(device)
        x['hand_image'] = torch.stack([
            default_image_transform(image)
            for a, image in zip(hand_activate, hand_color_render)
            if a
        ]).to(device)
        x['hand_cursor_activate'] = torch.BoolTensor(
            hand_activate).view(s, b).to(device)
        
        # process table tiles
        table_tiles, table_tyx, table_pad = batch_deduplicate_from_masks(
            observation['table_color_render'],
            observation['table_tile_mask'],
            observation['step'],
            pad,
        )
        x['table_tiles'] = default_tile_transform(
            table_tiles).to(device).contiguous()
        x['table_t'] = torch.LongTensor(table_tyx[...,0]).to(device)
        x['table_yx'] = torch.LongTensor(table_tyx[...,1:]).to(device)
        x['table_pad'] = torch.LongTensor(table_pad).to(device)
        
        # processs hand tiles
        hand_tiles, hand_tyx, hand_pad = batch_deduplicate_from_masks(
            observation['hand_color_render'],
            observation['hand_tile_mask'],
            observation['step'],
            pad,
        )
        x['hand_tiles'] = default_tile_transform(
            hand_tiles).to(device).contiguous()
        x['hand_t'] = torch.LongTensor(hand_tyx[...,0]).to(device)
        x['hand_yx'] = torch.LongTensor(hand_tyx[...,1:]).to(device)
        x['hand_pad'] = torch.LongTensor(hand_pad).to(device)
        
        # factor stuff, clean this up
        if self.config.factor_cursor_distribution:
            # this too needs to change if we properly combine these together
            # in the env/component later on.
            table_yx = observation['table_cursor']['position']
            table_yx = (
                table_yx[:,:,0] * self.config.table_decode_w + table_yx[:,:,1])
            x['table_cursor_yx'] = torch.LongTensor(table_yx).to(device)
            x['table_cursor_p'] = torch.LongTensor(
                observation['table_cursor']['polarity']).to(device)
            
            hand_yx = observation['hand_cursor']['position']
            hand_yx = (
                hand_yx[:,:,0] * self.config.hand_decode_w + hand_yx[:,:,1])
            x['hand_cursor_yx'] = torch.LongTensor(hand_yx).to(device)
            x['hand_cursor_p'] = torch.LongTensor(
                observation['hand_cursor']['polarity']).to(device)
        else:
            x['table_cursor_yx'] = None
            x['table_cursor_p'] = None
            x['hand_cursor_yx'] = None
            x['hand_cursor_p'] = None
        
        # process token x/t/pad
        phase_x = torch.LongTensor(observation['phase']).to(device)
        s, b = phase_x.shape
        decode_x = torch.full_like(phase_x, 2)
        x['token_x'] = torch.stack((phase_x, decode_x), dim=1).view(s*2,b)
        
        token_t = torch.LongTensor(observation['step']).to(device)
        x['token_t'] = torch.stack((token_t, token_t), dim=1).view(s*2,b)
        token_pad = torch.LongTensor(pad).to(device)
        x['token_pad'] = token_pad * 2
        
        # process decode t/pad
        # THIS COULD BE TOTALLY BAD.  ARE WE DECODING AT MULTIPLE SPOTS IN FACTORED AND THAT'S WHY WE NEED SMALLER BATCH SIZE???  Actually, it looks ok, the copying happens in the model, but we should double-check.  Oh but you know what, because the sequences are longer, that means our decode does get longer... I wonder if there's a way to tell it to only decode densely at certain locations, because we will only giving it a loss at those locations anyway.  This would mean making a 'table_decode_t' and a 'hand_decode_t' or something like that, and setting it equal to the locations where the hand or table is activated.  This would probably save a TON of memory in the normal setting too!  Yeah, definitely do this.  This could be huge for longer sequences too.
        #x['decode_t'] = token_t
        #x['decode_pad'] = token_pad
        
        #print('new')
        #for key, xx in x.items():
        #    if xx is not None:
        #        print(key, ':', xx.view(-1)[0].cpu())
        #print('table_tiles shape :', x['table_tiles'].shape)
        #print('hand_tiles shape :', x['hand_tiles'].shape)
        #print('table_color_render shape',
        #    observation['table_color_render'].shape)
        #print(numpy.sum(observation['table_color_render']))
        
        return x
    
    def augment(self, x, y):
        
        # get the batch size and device
        b = x['token_pad'].shape[0]
        device = x['token_pad'].device
        
        # apply tile shift augmentation
        max_shift = self.config.tile_shift_augmentation
        if max_shift:
            for region in 'table', 'hand':
                region_yx = x['%s_yx'%region]
                h = getattr(self.config, '%s_tiles_h'%region)
                w = getattr(self.config, '%s_tiles_w'%region)
                up_h = getattr(self.config, '%s_decode_h'%region) // h
                up_w = getattr(self.config, '%s_decode_w'%region) // w
                tile_shifts = []
                cursor_shifts = []
                for i in range(b):
                    
                    # compute shift min/max
                    p = x['%s_pad'%region][i]
                    min_shift_y = int(-torch.min(region_yx[:p,i,0]).cpu())
                    min_shift_y = max(min_shift_y, -max_shift)
                    max_shift_y = int((h-1)-torch.max(region_yx[:p,i,0]).cpu())
                    max_shift_y = min(max_shift_y, max_shift)
                    min_shift_x = int(-torch.min(region_yx[:p,i,1]).cpu())
                    min_shift_x = max(min_shift_x, -max_shift)
                    max_shift_x = int((w-1)-torch.max(region_yx[:p,i,1]).cpu())
                    max_shift_x = min(max_shift_x, max_shift)
                    
                    # pick shift randomly within min/max
                    shift_y = random.randint(min_shift_y, max_shift_y)
                    shift_x = random.randint(min_shift_x, max_shift_x)
                    tile_shifts.append([shift_y, shift_x])
                    cursor_shifts.append([shift_y * up_h, shift_x * up_w])
                
                # move shifts to torch/cuda
                tile_shifts = torch.LongTensor(tile_shifts).to(device)
                cursor_shifts = torch.LongTensor(cursor_shifts).to(device)
                
                # shift tile and cursor positions
                x['%s_yx'%region] = x['%s_yx'%region] + tile_shifts
                y['%s_yx'%region] = y['%s_yx'%region] + cursor_shifts
        
        # apply simulated misclicks
        if self.config.misclick_augmentation:
            max_t = max(
                torch.max(x['table_t']),
                torch.max(x['hand_t']),
                torch.max(x['token_t']),
            )
            for i in range(b):
                encode_shift_map = torch.arange(max_t+1)
                decode_shift_map = torch.arange(max_t+1)
                for j in range(max_t+1):
                    if random.random() < self.config.misclick_augmentation:
                        encode_shift_map[j+1:] += 1
                        decode_shift_map[j:] += 1
                
                # shift the table, hand, token and decode time steps
                x['table_t'][:,i] = encode_shift_map[x['table_t'][:,i]]
                x['hand_t'][:,i] = encode_shift_map[x['hand_t'][:,i]]
                x['token_t'][:,i] = encode_shift_map[x['token_t'][:,i]]
                #x['decode_t'][:,i] = decode_shift_map[x['decode_t'][:,i]]
        
        return x, y
    
    def forward_rollout(self, terminal, **x):
        device = x['table_tiles'].device
        use_memory = torch.BoolTensor(~terminal).to(device)
        return self.model(**x, use_memory=use_memory)
    
    def numpy_activations(self, x):
        a = {
            key:value.cpu().numpy().squeeze(axis=0)
            for key, value in x.items()
            if key not in ('table', 'hand', 'shape', 'color')
        }
        a['table'] = x['table'].cpu().numpy()
        a['hand'] = x['hand'].cpu().numpy()
        a['shape'] = x['shape'].cpu().numpy()
        a['color'] = x['color'].cpu().numpy()
        if len(a['shape'].shape) == 3:
            a['shape'] = a['shape'].squeeze(axis=0)
            a['color'] = a['color'].squeeze(axis=0)
        return a
