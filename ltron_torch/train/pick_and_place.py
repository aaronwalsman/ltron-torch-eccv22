#!/usr/bin/env python
import os
from collections import OrderedDict

import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import numpy

import PIL.Image as Image

import splendor.masks as masks

from ltron.visualization.drawing import (
    block_upscale_image, draw_crosshairs, line)
from ltron.hierarchy import len_hierarchy
from ltron.dataset.paths import get_dataset_paths

from ltron_torch.gym_tensor import default_image_transform
from ltron_torch.models.transformer import (
    Transformer,
    TransformerConfig,
)
from ltron_torch.models.transformer_masks import neighborhood
from ltron_torch.train.optimizer import OptimizerConfig, adamw_optimizer
import ltron_torch.models.dvae as dvae

def extract_path_indices(path, num_indices):
    parts = path.split('_')[-num_indices:]
    parts = tuple([int(part.split('.')[0]) for part in parts])
    return parts

def shorten_cache_data():

    print('loading dataset')
    cached_data = numpy.load(
        './dvae_cache.npz', allow_pickle=True)['arr_0'].item()
    
    print('generating shorter paths')
    paths = cached_data['frames'].keys()
    short_paths = []
    for path in tqdm.tqdm(paths):
        if 'color_x' in path:
            i, j = extract_path_indices(path, 2)
        elif 'color_y' in path:
            i, = extract_path_indices(path, 1)
        if i < 1000:
            short_paths.append(path)
    
    new_data = {}
    new_data['frames'] = {
        path:cached_data['frames'][path] for path in short_paths}
    new_data['snaps'] = {
        path:cached_data['snaps'][path] for path in short_paths}
    
    numpy.savez_compressed('./dvae_small_cache.npz', new_data)

class ImageDataset(Dataset):
    def __init__(self, dataset_name, split):
        subset = 1000
        paths = get_dataset_paths(dataset_name, split, subset=subset)
        self.frame_order = []
        for i in range(len_hierarchy(paths)):
            for j in range(7):
                self.frame_order.append(paths['color_x'][i].replace(
                    '_0000.png', '_%04i.png'%j))
        
        cached_data = numpy.load(
            '../dvae_small_cache.npz', allow_pickle=True)['arr_0'].item()
        self.data = cached_data
        
    def __len__(self):
        return len(self.frame_order)
    
    def __getitem__(self, index):
        x_path = self.frame_order[index]
        q_path = x_path.replace(
            'color_x_', 'color_y_')[:-9] + '.png'
        x = default_image_transform(Image.open(x_path))
        q = default_image_transform(Image.open(x_path))
        
        x_snaps = torch.zeros((8,3), dtype=torch.long)
        q_snaps = torch.zeros((8,3), dtype=torch.long)
        
        for i, snap in enumerate((
            (1,5),
            (1,6),
            (1,7),
            (1,8),
            (2,1),
            (2,2),
            (3,1),
            (3,2),
        )):
            if snap in self.data['snaps'][x_path]:
                yy, xx = self.data['snaps'][x_path][snap]
                x_snaps[i] = torch.LongTensor([yy,xx,1])
            else:
                x_snaps[i] = torch.LongTensor([0,0,0])
            
            if snap in self.data['snaps'][q_path]:
                yy, xx = self.data['snaps'][q_path][snap]
                q_snaps[i] = torch.LongTensor([yy,xx,1])
            else:
                q_snaps[i] = torch.LongTensor([0,0,0])
        
        return x, q, x_snaps, q_snaps, x_path, q_path

class CachedDataset(Dataset):
    def __init__(self, dataset_name, split):
        
        # ----------------------------------------------------------------------
        print('loading dataset')
        cached_data = numpy.load(
            '../dvae_small_cache.npz', allow_pickle=True)['arr_0'].item()
        subset=1000
        # ----------------------------------------------------------------------
        #print('loading dataset')
        #cached_data = numpy.load(
        #    './dvae_cache.npz', allow_pickle=True)['arr_0'].item()
        #subset=None
        # ----------------------------------------------------------------------
    
        paths = get_dataset_paths(dataset_name, split, subset=subset)
        self.frame_order = []
        for i in range(len_hierarchy(paths)):
            for j in range(7):
                self.frame_order.append(paths['color_x'][i].replace(
                    '_0000.png', '_%04i.png'%j))
        
        self.data = cached_data
    
    def __len__(self):
        return len(self.frame_order)
    
    def __getitem__(self, index):
        x_path = self.frame_order[index]
        q_path = x_path.replace(
            'color_x_', 'color_y_')[:-9] + '.png'
        x = self.data['frames'][x_path]
        q = self.data['frames'][q_path]
        
        x_snaps = torch.zeros((8,3), dtype=torch.long)
        q_snaps = torch.zeros((8,3), dtype=torch.long)
        
        for i, snap in enumerate((
            (1,5),
            (1,6),
            (1,7),
            (1,8),
            (2,1),
            (2,2),
            (3,1),
            (3,2),
        )):
            if snap in self.data['snaps'][x_path]:
                yy, xx = self.data['snaps'][x_path][snap]
                x_snaps[i] = torch.LongTensor([yy,xx,1])
            else:
                x_snaps[i] = torch.LongTensor([0,0,0])
            
            if snap in self.data['snaps'][q_path]:
                yy, xx = self.data['snaps'][q_path][snap]
                q_snaps[i] = torch.LongTensor([yy,xx,1])
            else:
                q_snaps[i] = torch.LongTensor([0,0,0])
        
        return x, q, x_snaps, q_snaps, x_path, q_path


def visualize():
    print('making dataset')
    dataset = CachedDataset(
        'conditional_snap_two_frames',
        'train',
    )
    
    for i, (x, q, snaps, path) in enumerate(dataset):
        image = Image.open(path)
        
        blocks = masks.color_index_to_byte(x)
        blocks = block_upscale_image(blocks, 256, 256)
        
        image = numpy.concatenate((image, blocks), axis=1)
        out_path = os.path.basename(path).replace('color_x_', 'vis_')
        Image.fromarray(image).save(out_path)
        
        if i > 16:
            break

# Dense Pick ===================================================================

class DensePickModel(torch.nn.Module):
    def __init__(self, channels=1024):
        super(DensePickModel, self).__init__()
        #self.decoder = ImageSequenceEncoder(
        #    tokens_per_image=32*32,
        #    max_seq_length=1,
        #    n_read_tokens=0,
        #    read_channels=9,
        #    read_from_input=True,
        #    channels=channels,
        #    num_layers=12,
        #)
        config = TransformerConfig(
            decoder_channels = 9,
            num_blocks=6,
            channels=512,
            num_heads=8,
        )
        self.decoder = Transformer(config)
    
    def forward(self, x):
        return self.decoder(x)

class DenseConvModel(torch.nn.Module):
    def __init__(self):
        super(DenseConvModel, self).__init__()
        self.embedding = torch.nn.Embedding(4096, 128)
        num_groups = 4
        blocks_per_group = 2
        num_layers = num_groups * blocks_per_group
        groups = OrderedDict()
        groups['group_1'] = dvae.DecoderGroup(
            128, 256*8, blocks_per_group, num_layers, upsample=False)
        groups['group_2'] = dvae.DecoderGroup(
            256*8, 256*4, blocks_per_group, num_layers, upsample=False)
        groups['group_3'] = dvae.DecoderGroup(
            256*4, 256*2, blocks_per_group, num_layers, upsample=False)
        groups['group_4'] = dvae.DecoderGroup(
            256*2, 256, blocks_per_group, num_layers, upsample=False)
        groups['out'] = torch.nn.Sequential(OrderedDict([
            ('relu', torch.nn.ReLU()),
            ('conv', torch.nn.Conv2d(256, 9, kernel_size=1)),
        ]))
        
        self.groups = torch.nn.Sequential(groups)
    
    def forward(self, x):
        x = self.embedding(x)
        s, hw, b, c = x.shape
        x = x.view(hw, b, c).permute(1,2,0).reshape(b,c,32,32)
        x = self.groups(x)
        b,c,h,w = x.shape
        x = x.view(b,c,h*w).permute(2,0,1).contiguous()
        
        return x

def train_dense_pick(
    num_epochs=100,
):
    print('making dataset')
    train_dataset = CachedDataset(
        'conditional_snap_two_frames',
        'train',
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
    )
    
    test_dataset = CachedDataset(
        'conditional_snap_two_frames',
        'train',
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
    )
    
    print('making model')
    model = DensePickModel().cuda()
    #model = DenseConvModel().cuda()
    #model.load_state_dict(torch.load('./dense_02/model_0050.pt'))
    
    #attention_mask = neighborhood(32, 32, width=3).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    running_snap_loss = 0.
    for epoch in range(1, num_epochs+1):
        print('epoch: %i'%epoch)
        print('train')
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for x, q, x_snaps, q_snaps, x_paths, q_paths in iterate:
            x = x.cuda()
            
            b, h, w = x.shape
            x = x.view(b, h*w).permute(1,0).unsqueeze(0)
            
            x_snaps = x_snaps.permute(1, 0, 2).contiguous().cuda()
            #snap_locations = (snaps[:,:,:2] - 31.5) / 31.5
            snap_locations = x_snaps[:,:,:2]
            snap_valid = x_snaps[:,:,[2]]#.view(8*b)
            
            #x = model(x, mask=attention_mask)
            x = model(x)
            
            total_loss = 0.
            
            snap_target = torch.zeros(32*32, b, dtype=torch.long).cuda()
            for i in range(8):
                yx = snap_locations[i]
                yy = (yx[:,0] / 2.).long()
                xx = (yx[:,1] / 2.).long()
                yx = yy * 32 + xx
                snap_target[yx, range(b)] = i + 1
            
            class_weight = torch.ones(9)
            class_weight[0] = 0.1
            
            snap_loss = torch.nn.functional.cross_entropy(
                x.view(32*32*b, -1),
                snap_target.view(32*32*b),
                weight=class_weight.cuda()
            )
            
            total_loss = total_loss + snap_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_snap_loss = running_snap_loss * 0.9 + float(snap_loss) * 0.1
            iterate.set_description('s: %.04f'%float(running_snap_loss))
        
        torch.save(model.state_dict(), 'model_%04i.pt'%epoch)
        
        with torch.no_grad():
            model.eval()
            iterate = tqdm.tqdm(test_loader)
            for x, q, x_snaps, q_snaps, x_paths, q_paths in iterate:
                x = x.cuda()
                
                b, h, w = x.shape
                x = x.view(b, h*w).permute(1,0).unsqueeze(0)
                
                x = model(x)
                
                #x = torch.softmax(x, dim=-1)
                #x = x.view(8, b, 64, 64)
                #snap_maps = torch.sum(x, dim=0).unsqueeze(-1).cpu().numpy()
                
                x_snaps = x_snaps.permute(1, 0, 2).contiguous().cuda()
                snap_locations = x_snaps[:,:,:2]
                
                for i in range(b):
                    path = x_paths[i]
                    image = numpy.array(Image.open(path))
                    
                    #x_map = torch.softmax(x[:,i], dim=-1).cpu().numpy()
                    
                    x_class = torch.argmax(x[:,i], dim=-1).cpu().numpy()
                    x_class = x_class.reshape(32, 32)
                    x_background = x_class == 0
                    class_color = masks.color_index_to_byte(x_class)
                    class_color = block_upscale_image(class_color, 256, 256)
                    x_background = block_upscale_image(x_background, 256, 256)
                    x_background = x_background.reshape(256, 256, 1)
                    image = (
                        image * x_background +
                        class_color * (1. - x_background)
                    ).astype(numpy.uint8)
                    
                    #p_normal = x_map[:,0].reshape(32, 32, 1)
                    #p_normal = block_upscale_image(p_normal, 256, 256)
                    #image = image * p_normal + (1. - p_normal) * [0,0,255]
                    image = image.astype(numpy.uint8)
                    
                    '''
                    #snap_map = snap_maps[i]
                    #snap_map = block_upscale_image(snap_map, 256, 256)
                    #image = image * (1. - snap_map) + [0,0,255] * snap_map
                    
                    neg_path = path.replace(
                        'color_x_', 'snap_neg_').replace(
                        '.png', '.npz')
                    neg_snap_map = numpy.load(neg_path, allow_pickle=True)
                    neg_snap_map_i = neg_snap_map['arr_0'][:,:,0]
                    #neg_snap_map_s = neg_snap_map['arr_0'][:,:,1]
                    neg_snap_map = (neg_snap_map_i == 2) | (neg_snap_map_i == 3)
                    
                    pos_path = path.replace(
                        'color_x_', 'snap_pos_').replace(
                        '.png', '.npz')
                    pos_snap_map = numpy.load(pos_path, allow_pickle=True)
                    pos_snap_map_i = pos_snap_map['arr_0'][:,:,0]
                    #pos_snap_map_s = pos_snap_map['arr_0'][:,:,1]
                    pos_snap_map = pos_snap_map_i == 1
                    
                    gt_dense_snap = neg_snap_map | pos_snap_map
                    gt_dense_snap = gt_dense_snap.reshape(64, 64, 1)
                    gt_dense_snap = block_upscale_image(
                        gt_dense_snap, 256, 256)
                    gt_dense_snap = gt_dense_snap * 0.5
                    image = (
                        image * (1. - gt_dense_snap) +
                        [0,255,0] * gt_dense_snap
                    )
                    
                    gt_snaps = numpy.zeros(4096)
                    gt_snaps[snap_locations[:,i].cpu()] = 1
                    gt_snaps = gt_snaps.reshape(64, 64, 1)
                    gt_snaps = block_upscale_image(gt_snaps, 256, 256)
                    gt_snaps = gt_snaps * 0.5
                    image = image * (1. - gt_snaps) + [255,0,255] * gt_snaps
                    '''
                    image = image.astype(numpy.uint8)
                    Image.fromarray(image).save(
                        './tmp_%04i_%04i.png'%(epoch, i))
                
                break

# Sparse Pick ==================================================================

def train_sparse_pick(
    num_epochs=100,
):
    print('making dataset')
    train_dataset = CachedDataset(
        'conditional_snap_two_frames',
        'train',
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
    )
    
    test_dataset = CachedDataset(
        'conditional_snap_two_frames',
        'train',
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
    )
    
    print('making model')
    config = TransformerConfig(
        decoder_tokens = 8,
        decode_input = False,
        
        decoder_channels = 2,
        num_blocks=6,
        channels=512,
        num_heads=8,
    )
    model = Transformer(config).cuda()
    
    optimizer_config = OptimizerConfig()
    optimizer = adamw_optimizer(model, optimizer_config)
    
    running_loss = 0.
    for epoch in range(1, num_epochs+1):
        print('epoch: %i'%epoch)
        print('train')
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for x, q, x_snaps, q_snaps, x_paths, q_paths in iterate:
            x = x.cuda()
            
            b, h, w = x.shape
            x = x.view(b, h*w).permute(1,0).unsqueeze(0)
            
            x_snaps = x_snaps.permute(1, 0, 2).contiguous().cuda()
            snap_locations = (x_snaps[:,:,:2] - 31.5) #/ 31.5
            snap_valid = x_snaps[:,:,[2]]#.view(8*b)
            snap_targets = snap_locations
            
            x = model(x)
            
            total_loss = 0.
            
            snap_loss = torch.nn.functional.smooth_l1_loss(x, snap_targets)
            total_loss = total_loss + snap_loss # * 0.001
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss = running_loss * 0.9 + float(snap_loss) * 0.1
            iterate.set_description('s: %.04f'%running_loss)
        
        torch.save(model.state_dict(), 'model_%04i.pt'%epoch)
        
        with torch.no_grad():
            model.eval()
            iterate = tqdm.tqdm(test_loader)
            for x, q, x_snaps, q_snaps, x_paths, q_paths in iterate:
                x = x.cuda()
                
                b, h, w = x.shape
                x = x.view(b, h*w).permute(1,0).unsqueeze(0)
                
                x = model(x)
                
                x_snaps = x_snaps.permute(1, 0, 2).contiguous().cuda()
                snap_locations = x_snaps[:,:,0] * 64 + x_snaps[:,:,1]
                
                for i in range(b):
                    path = x_paths[i]
                    image = numpy.array(Image.open(path))
                    
                    neg_path = path.replace(
                        'color_x_', 'snap_neg_').replace(
                        '.png', '.npz')
                    neg_snap_map = numpy.load(neg_path, allow_pickle=True)
                    neg_snap_map_i = neg_snap_map['arr_0'][:,:,0]
                    neg_snap_map = (neg_snap_map_i == 2) | (neg_snap_map_i == 3)
                    
                    pos_path = path.replace(
                        'color_x_', 'snap_pos_').replace(
                        '.png', '.npz')
                    pos_snap_map = numpy.load(pos_path, allow_pickle=True)
                    pos_snap_map_i = pos_snap_map['arr_0'][:,:,0]
                    pos_snap_map = pos_snap_map_i == 1
                    
                    gt_dense_snap = neg_snap_map | pos_snap_map
                    gt_dense_snap = gt_dense_snap.reshape(64, 64, 1)
                    gt_dense_snap = block_upscale_image(
                        gt_dense_snap, 256, 256)
                    gt_dense_snap = gt_dense_snap * 0.5
                    image = (
                        image * (1. - gt_dense_snap) +
                        [0,255,0] * gt_dense_snap
                    )
                    
                    gt_snaps = numpy.zeros(4096)
                    gt_snaps[snap_locations[:,i].cpu()] = 1
                    gt_snaps = gt_snaps.reshape(64, 64, 1)
                    gt_snaps = block_upscale_image(gt_snaps, 256, 256)
                    gt_snaps = gt_snaps * 0.5
                    image = image * (1. - gt_snaps) + [255,0,255] * gt_snaps
                    
                    for j in range(8):
                        yy, xx = x[j, i].cpu().numpy()
                        yy = yy * 4 + 127.5
                        xx = xx * 4 + 127.5
                        draw_crosshairs(image, xx, yy, 5, [0, 0, 255])
                    
                    image = image.astype(numpy.uint8)
                    Image.fromarray(image).save(
                        './tmp_%04i_%04i.png'%(epoch, i))
                
                break

# Sparse Pick And Place ========================================================

def train_sparse_pick_and_place(
    input_type='tokens',
    num_epochs=100,
):
    
    print('making dataset')
    print('TESTING ON TRAIN SET')
    if input_type == 'tokens':
        train_dataset = CachedDataset(
            'conditional_snap_two_frames',
            'train',
        )
        test_dataset = CachedDataset(
            'conditional_snap_two_frames',
            'train',
        )
    elif input_type == 'images':
        train_dataset = ImageDataset(
            'conditional_snap_two_frames',
            'train',
        )
        test_dataset = ImageDataset(
            'conditional_snap_two_frames',
            'train',
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    
    print('making model')
    config = TransformerConfig(
        t = 2,
        decoder_tokens = 8,
        decode_input = False,
        
        input_type = input_type,
        
        decoder_channels = 4,
        num_blocks=6,
        channels=256,
        num_heads=8,
        residual_channels=256,
    )
    model = Transformer(config).cuda()
    
    optimizer_config = OptimizerConfig()
    optimizer = adamw_optimizer(model, optimizer_config)
    
    running_loss = 0.
    for epoch in range(1, num_epochs+1):
        print('epoch: %i'%epoch)
        print('train')
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for x, q, x_snaps, q_snaps, x_paths, q_paths in iterate:
            x = x.cuda()
            q = q.cuda()
            
            if input_type == 'tokens':
                b, h, w = x.shape
                x = x.view(b, h*w).permute(1,0)
                q = q.view(b, h*w).permute(1,0)
                qx = torch.stack((q,x), dim=0)
            elif input_type == 'images':
                qx = torch.stack((q, x), dim=-3)
            
            qx = model(qx)
            
            total_loss = 0.
            
            x_snaps = x_snaps.permute(1, 0, 2).contiguous().cuda()
            x_snap_locations = (x_snaps[:,:,:2] - 31.5) #/ 31.5
            x_snap_valid = x_snaps[:,:,[2]]#.view(8*b)
            
            q_snaps = q_snaps.permute(1, 0, 2).contiguous().cuda()
            q_snap_locations = (q_snaps[:,:,:2] - 31.5)
            q_snap_valid = q_snaps[:,:,[2]]
            
            snap_targets = torch.cat(
                (q_snap_locations, x_snap_locations), dim=-1)
            
            snap_loss = torch.nn.functional.smooth_l1_loss(qx, snap_targets)
            total_loss = total_loss + snap_loss # * 0.001
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss = running_loss * 0.9 + float(snap_loss) * 0.1
            iterate.set_description('s: %.04f'%running_loss)
        
        torch.save(model.state_dict(), 'model_%04i.pt'%epoch)
        
        with torch.no_grad():
            model.eval()
            iterate = tqdm.tqdm(test_loader)
            n = 0
            for x, q, x_snaps, q_snaps, x_paths, q_paths in iterate:
                x = x.cuda()
                q = q.cuda()
                
                if input_type == 'tokens':
                    b, h, w = x.shape
                    x = x.view(b, h*w).permute(1,0)
                    q = q.view(b, h*w).permute(1,0)
                    qx = torch.stack((q,x), dim=0)
                elif input_type == 'images':
                    qx = torch.stack((q, x), dim=-3)
                    b, c, t, h, w = qx.shape
                
                qx = model(qx)
                
                x_snaps = x_snaps.permute(1, 0, 2).contiguous().cuda()
                snap_locations = x_snaps[:,:,0] * 64 + x_snaps[:,:,1]
                
                for i in range(b):
                    x_path = x_paths[i]
                    x_image = numpy.array(Image.open(x_path))
                    
                    q_path = q_paths[i]
                    q_image = numpy.array(Image.open(q_path))
                    
                    neg_path = x_path.replace(
                        'color_x_', 'snap_neg_').replace(
                        '.png', '.npz')
                    neg_snap_map = numpy.load(neg_path, allow_pickle=True)
                    neg_snap_map_i = neg_snap_map['arr_0'][:,:,0]
                    neg_snap_map = (neg_snap_map_i == 2) | (neg_snap_map_i == 3)
                    
                    pos_path = x_path.replace(
                        'color_x_', 'snap_pos_').replace(
                        '.png', '.npz')
                    pos_snap_map = numpy.load(pos_path, allow_pickle=True)
                    pos_snap_map_i = pos_snap_map['arr_0'][:,:,0]
                    pos_snap_map = pos_snap_map_i == 1
                    
                    gt_dense_snap = neg_snap_map | pos_snap_map
                    gt_dense_snap = gt_dense_snap.reshape(64, 64, 1)
                    gt_dense_snap = block_upscale_image(
                        gt_dense_snap, 256, 256)
                    gt_dense_snap = gt_dense_snap * 0.5
                    x_image = (
                        x_image * (1. - gt_dense_snap) +
                        [0,255,0] * gt_dense_snap
                    )
                    
                    gt_snaps = numpy.zeros(4096)
                    gt_snaps[snap_locations[:,i].cpu()] = 1
                    gt_snaps = gt_snaps.reshape(64, 64, 1)
                    gt_snaps = block_upscale_image(gt_snaps, 256, 256)
                    gt_snaps = gt_snaps * 0.5
                    x_image = x_image * (1. - gt_snaps) + [255,0,255] * gt_snaps
                    
                    for j in range(8):
                        yy1, xx1 = qx[j, i, :2].cpu().numpy()
                        yy1 = round(yy1 * 4 + 127.5)
                        xx1 = round(xx1 * 4 + 127.5)
                        yy2, xx2 = qx[j, i, 2:].cpu().numpy()
                        yy2 = round(yy2 * 4 + 127.5)
                        xx2 = round(xx2 * 4 + 127.5)
                        yy, xx = line(yy1, xx1, yy2, xx2)
                        x_image[yy,xx] = [0,0,255]
                        #draw_crosshairs(image, xx, yy, 5, [0, 0, 255])
                    
                    x_image = x_image.astype(numpy.uint8)
                    image = numpy.concatenate((q_image, x_image), axis=1)
                    Image.fromarray(image).save(
                        './tmp_%04i_%04i.png'%(epoch, n))
                    n += 1
                
                if n > 16:
                    break

if __name__ == '__main__':
    #shorten_cache_data()
    #visualize()
    #train_dense_pick()
    #train_sparse_pick()
    train_sparse_pick_and_place(input_type='images')
