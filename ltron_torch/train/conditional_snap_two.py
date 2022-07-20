#!/usr/bin/env python
import os

import numpy

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import tqdm

from ltron.hierarchy import len_hierarchy
from ltron.dataset.paths import get_dataset_paths
from ltron.visualization import drawing

from ltron_torch.gym_tensor import (
    default_image_transform, default_image_untransform)
#from ltron_torch.models.positional_encoding import raw_positional_encoding
from ltron_torch.models.sequence_fcn import (
    named_resnet_independent_sequence_fcn)

class SeqDataset(Dataset):
    def __init__(
        self,
        dataset,
        split,
        subset=None,
        seq_len=7
    ):
        self.dataset_paths = get_dataset_paths(dataset, split, subset)
        self.seq_len = seq_len
    
    def __len__(self):
        return len_hierarchy(self.dataset_paths)
    
    def __getitem__(self, index):
        color_x_path = self.dataset_paths['color_x'][index]
        color_y_path = self.dataset_paths['color_y'][index]
        snap_neg_path = self.dataset_paths['snap_neg'][index]
        snap_pos_path = self.dataset_paths['snap_pos'][index]
        action_path = self.dataset_paths['action'][index]
        target_graph_path = self.dataset_paths['target_graph'][index]
        poses_y_path = self.dataset_paths['poses_y'][index]
        
        color_xs = []
        snap_negs = []
        snap_poss = []
        actions = []
        for i in range(self.seq_len):
            color_x_i_path = color_x_path.replace('_0000.png', '_%04i.png'%i)
            color_x_image = Image.open(color_x_i_path)
            color_x = default_image_transform(color_x_image)
            color_xs.append(color_x)
            
            snap_neg_i_path = snap_neg_path.replace('_0000.npy', '_%04i.npy'%i)
            snap_neg = numpy.load(snap_neg_i_path)['arr_0']
            snap_neg = numpy.moveaxis(snap_neg, -1, -3)
            snap_neg = torch.LongTensor(snap_neg)
            snap_negs.append(snap_neg)
            
            snap_pos_i_path = snap_pos_path.replace('_0000.npy', '_%04i.npy'%i)
            snap_pos = numpy.load(snap_pos_i_path)['arr_0']
            snap_pos = numpy.moveaxis(snap_pos, -1, -3)
            snap_pos = torch.LongTensor(snap_pos)
            snap_poss.append(snap_pos)
            
            action_i_path = action_path.replace('_0000.npy', '_%04i.npy'%i)
            action = numpy.load(action_i_path, allow_pickle=True).item()
            torch_action = torch.zeros(7, dtype=numpy.long)
            torch_action[0:2] = torch.LongTensor(
                action['pick_and_place']['pick'])
            torch_action[2:4] = torch.LongTensor(
                action['pick_and_place']['place'])
            torch_action[4:6] = torch.LongTensor(
                action['vector_offset']['pick'])
            torch_action[6] = action['vector_offset']['direction'][1] == 1
            actions.append(torch_action)
        
        color_x = torch.stack(color_xs, dim=0)
        
        color_y_image = Image.open(color_y_path)
        color_y = default_image_transform(color_y_image)
        
        snap_neg = torch.stack(snap_negs, dim=0)
        snap_pos = torch.stack(snap_poss, dim=0)
        
        actions = torch.stack(actions, dim=0)
        
        poses_y = torch.FloatTensor(numpy.load(poses_y_path))
        
        return color_x, color_y, snap_neg, snap_pos, actions, poses_y

def seq_collate(frames):
    color_x = [f[0] for f in frames]
    color_x = torch.stack(color_x, dim=1)
    color_y = [f[1] for f in frames]
    color_y = torch.stack(color_y, dim=0).unsqueeze(0)
    snap_neg = [f[2] for f in frames]
    snap_neg = torch.stack(snap_neg, dim=1)
    snap_pos = [f[3] for f in frames]
    snap_pos = torch.stack(snap_pos, dim=1)
    actions = [f[4] for f in frames]
    actions = torch.stack(actions, dim=1)
    poses_y = [f[5] for f in frames]
    poses_y = torch.stack(poses_y, dim=0).unsqueeze(0)
    
    return color_x, color_y, snap_neg, snap_pos, actions, poses_y

class SeqModel(torch.nn.Module):
    def __init__(
        self,
        channels=256,
        image_block_size=16,
        transformer_layers=6,
        transformer_heads=4,
        transformer_dropout=0.5,
        embedding_dropout=0.1,
        max_seq_len=8,
        max_spatial_len=256,
        output_cells=4,
    ):
        super(SeqModel, self).__init__()
        self.channels=channels
        self.image_block_size = image_block_size
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_dropout = transformer_dropout
        
        self.register_buffer(
            'sequence_encoding',
            raw_positional_encoding(channels, max_seq_len),
        )
        self.sequence_linear = torch.nn.Linear(channels, channels)
        self.sequence_dropout = torch.nn.Dropout(embedding_dropout)
        self.register_buffer(
            'spatial_encoding',
            raw_positional_encoding(channels, max_spatial_len),
        )
        self.spatial_linear = torch.nn.Linear(channels, channels)
        self.spatial_dropout = torch.nn.Dropout(embedding_dropout)
        
        self.block_encoder = torch.nn.Conv2d(
            3,
            channels,
            kernel_size=image_block_size,
            stride=image_block_size,
            padding=0,
        )
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            self.transformer_heads,
            channels,
            self.transformer_dropout,
        )
        self.transformer_decoder = torch.nn.TransformerEncoder(
            transformer_layer, self.transformer_layers)
        
        self.out_linear = torch.nn.Linear(channels, (4+64+64)*4*4)
    
    def forward(self, x, q, padding_mask=None):
        # add the target to the start of the sequence
        qx = torch.cat((q,x), dim=0)
        
        # run the conv to get spatial blocks
        s, b, c, h, w = qx.shape
        qx = qx.view(s*b, c, h, w)
        sb, c, h, w = qx.shape
        qx = self.block_encoder(qx)
        
        # add sequence encoding to qx
        sb, c, h, w = qx.shape
        qx = qx.view(s, b, c, h*w)
        qx = qx.permute(0, 3, 1, 2)
        s_e = self.sequence_linear(self.sequence_encoding[:s])
        s_e = self.sequence_dropout(s_e)
        s_e = s_e.view(s, 1, 1, c)
        qx = qx + s_e
        
        # add spatial encoding to qx
        hw_e = self.spatial_linear(self.spatial_encoding[:h*w])
        hw_e = self.spatial_dropout(hw_e)
        hw_e = hw_e.view(1, h*w, 1, c)
        qx = qx + hw_e
        
        # transformer time
        qx = qx.reshape(s*h*w, b, c)
        
        # make the masks
        seq_mask = (torch.triu(torch.ones(s,s)) == 1).transpose(0,1)
        seq_mask = seq_mask.float().masked_fill(
            seq_mask == 0, float('-inf')).masked_fill(
            seq_mask == 1, float(0.))
        seq_mask = seq_mask.unsqueeze(1).expand(s,h*w,s)
        seq_mask = seq_mask.unsqueeze(3).expand(s,h*w,s,h*w)
        seq_mask = seq_mask.reshape(s*h*w, s*h*w)
        seq_mask = seq_mask.to(qx.device)
        
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(2).expand(b,s,h*w)
            padding_mask = padding_mask.reshape(b,s*h*w)
        
        qx = self.transformer_decoder(
            qx,
            mask=seq_mask,
            src_key_padding_mask=padding_mask
        )
        
        qx = self.out_linear(qx)
        shw, b, c = qx.shape
        qx = qx.view(s, h, w, b, (4+64+64), 4, 4)
        qx = qx.permute(0, 3, 4, 1, 5, 2, 6)
        qx = qx.reshape(s, b, (4+64+64), h*4, w*4)
        
        return qx

def mode_targets_from_actions(actions, snap_neg, snap_pos, device):
    '''
    modes:
    0 - no-op
    1 - drag-and-drop
    2 - rotate by (0,-1, 0)
    3 - rotate by (0, 1, 0)
    '''
    s, b = actions.shape[:2]
    mode_targets = torch.zeros((s, b, 64, 64), dtype=torch.long)
    mode_targets = mode_targets.to(device)
    
    def set_mode_targets_from_snap(frames, snap, value):
        snap = snap.unsqueeze(-1).unsqueeze(-1)
        snap_maps = snap_neg[frames[:,0], frames[:,1]] == snap
        snap_map = snap_maps[:,0] & snap_maps[:,1]
        
        mode_targets[frames[:,0], frames[:,1]] = snap_map.long() * value
    
    # set pick and place mode targets
    p_and_p_frames = torch.nonzero(actions[:,:,0], as_tuple=False)
    p_and_p_actions = actions[p_and_p_frames[:,0], p_and_p_frames[:,1]]
    pick_instance_snap = p_and_p_actions[:,:2]
    set_mode_targets_from_snap(p_and_p_frames, pick_instance_snap, 1)
            
    # set rotate targets
    for rotate_direction in 0,1:
        rotate_frames = torch.nonzero(
            (actions[:,:,4] != 0) &
            (actions[:,:,6] == rotate_direction),
            as_tuple=False
        )
        rotate_actions = actions[
            rotate_frames[:,0],
            rotate_frames[:,1],
        ]
        rotate_instance_snap = rotate_actions[:,4:6]
        set_mode_targets_from_snap(
            rotate_frames, rotate_instance_snap, rotate_direction+2)
    
    return mode_targets
            
def train(
    epochs=50,
    batch_size=4,
    train_subset=None,
    test_subset=None,
    seq_len=None,
):
    
    train_dataset = SeqDataset(
        'conditional_snap_two_frames',
        'train',
        subset=train_subset,
        seq_len=seq_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=seq_collate,
    )
    
    test_dataset = SeqDataset(
        'conditional_snap_two_frames',
        'test',
        subset=test_subset,
        seq_len=seq_len,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=seq_collate,
    )
    
    #model = SeqModel().cuda()
    model = named_resnet_independent_sequence_fcn(
        'resnet50', (4+64+64), pretrained=True).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    class_weight = torch.FloatTensor([0.01, 1, 1, 1]).cuda()
    
    for epoch in range(1, epochs+1):
        print('='*80)
        print('Epoch: %i'%epoch)
        print('-'*80)
        print('Train')
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for color_x, color_y, snap_neg, snap_pos, actions, poses_y in iterate:
            color_x = color_x.cuda()
            color_y = color_y.cuda()
            snap_neg = snap_neg.cuda()
            snap_pos = snap_pos.cuda()
            actions = actions.cuda()
            poses_y = poses_y.cuda()
            
            #action_logits = model(color_x, color_y)[1:]
            action_logits = model(color_x)
            mode_logits = action_logits[:,:,:4]
            p_and_p_logits = action_logits[:,:,4:]
            
            # build action targets
            s, b, c, h, w = action_logits.shape
            
            # supervise
            total_loss = 0
            
            mode_targets = mode_targets_from_actions(
                actions, snap_neg, snap_pos, action_logits.device)
            mode_loss = torch.nn.functional.cross_entropy(
                mode_logits.view(s*b,4,h,w),
                mode_targets.view(s*b,h,w),
                class_weight)
            
            total_loss = total_loss + mode_loss
            
            iterate.set_description('m:%.03f'%float(mode_loss))
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print('-'*80)
        print('Test')
        model.eval()
        with torch.no_grad():
            iterate = tqdm.tqdm(test_loader)
            for (color_x,
                color_y,
                snap_neg,
                snap_pos,
                actions,
                poses_y,
            ) in iterate:
                color_x = color_x.cuda()
                color_y = color_y.cuda()
                snap_neg = snap_neg.cuda()
                snap_pos = snap_pos.cuda()
                actions = actions.cuda()
                poses_y = poses_y.cuda()
                
                #action_logits = model(color_x, color_y)[1:]
                action_logits = model(color_x)
                mode_logits = action_logits[:,:,:4]
                p_mode = torch.softmax(mode_logits, dim=2)
                p_and_p_logits = action_logits[:,:,4:]
                
                mode_targets = mode_targets_from_actions(
                    actions, snap_neg, snap_pos, action_logits.device)
                
                full_h, full_w = color_x.shape[-2:]
                s, b, c, h, w = mode_logits.shape
                for i in range(b):
                    color_y_image = default_image_untransform(color_y[0,i])
                    Image.fromarray(color_y_image).save('./color_y_%i.png'%i)
                    for j in range(s):
                        color_x_image = default_image_untransform(color_x[j,i])
                        
                        p_action = p_mode[j,i,1:].permute(1,2,0).cpu().numpy()
                        p_action = drawing.block_upscale_image(
                            p_action, full_w, full_h)
                        p_pick = numpy.sum(p_action, axis=-1, keepdims=True)
                        p_pick = p_pick * 0.5
                        p_color = 1. - p_pick
                        
                        out = color_x_image * p_color + p_action * 255 * p_pick
                        out = out.astype(numpy.uint8)
                        Image.fromarray(out).save('./color_x_%i_%i.png'%(i,j))
                        
                        target = mode_targets[j,i]
                        translate_target = target == 1
                        rot_neg_target = target == 2
                        rot_pos_target = target == 3
                        target_rgb = torch.stack(
                            (translate_target, rot_neg_target, rot_pos_target),
                            dim=-1).cpu().numpy()
                        target_rgb = drawing.block_upscale_image(
                            target_rgb, full_w, full_h)
                        p_target = numpy.sum(target_rgb, axis=-1, keepdims=True)
                        p_target = p_target * 0.5
                        p_color = 1. - p_target
                        
                        out = color_x_image * p_color + (
                            target_rgb * p_target * 255)
                        out = out.astype(numpy.uint8)
                        Image.fromarray(out).save('./target_x_%i_%i.png'%(i,j))
                break
        
        torch.save(model.state_dict(), './model_%04i.pt'%epoch)

if __name__ == '__main__':
    train(train_subset=None, seq_len=1)
