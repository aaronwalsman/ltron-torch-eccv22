#!/usr/bin/env python
import math
import os

import tqdm

import PIL.Image as Image

import numpy

import torch
from torch.utils.data import Dataset, DataLoader

from ltron.dataset.paths import get_dataset_paths

import ltron_torch.models.dvae as dvae
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

def cosine_anneal(start_value, end_value, step, max_steps):
    if step < max_steps:
        t = step / max_steps
        return (
            end_value +
            (start_value - end_value) * 0.5 * (1. + math.cos(t * math.pi))
        )
    else:
        return end_value

def linear_anneal(start_value, end_value, step, max_steps):
    if step < max_steps:
        t = step / max_steps
        return start_value + (end_value - start_value) * t
    else:
        return end_value

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
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    model = dvae.DVAE(
        hidden_channels=64,
        output_channels=(256*3),
        vocabulary_size=4096,
    ).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for epoch in range(1, epochs+1):
        print('Epoch: %i'%epoch)
        print('Train:')
        model.train()
        tau = cosine_anneal(1., 1/16., epoch-1, 50)
        kl_loss_weight = linear_anneal(0., 0.0001, epoch-1, 50)
        model.update_tau(tau)
        iterate = tqdm.tqdm(train_loader)
        for x, y, p in iterate:
            x = x.cuda()
            y = y.cuda()
            
            z, z_sample, x = model(x)
            
            total_loss = 0
            
            kl_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(z, dim=1),
                torch.ones_like(z) * (1 / z.shape[1]),
                reduction='batchmean',
            ) * kl_loss_weight
            
            # this was not done in the first run
            total_loss = total_loss + kl_loss
            
            b, c, h, w = x.shape
            x = x.view(b, c//3, 3, h, w)
            reconstruction_loss = torch.nn.functional.cross_entropy(x, y)
            
            total_loss = total_loss + reconstruction_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            iterate.set_description(
                'r:%.04f kl:%.04f'%(float(reconstruction_loss), float(kl_loss)))
        
        torch.save(model.state_dict(), './model_%04i.pt'%epoch)
        
        model.eval()
        with torch.no_grad():
            iterate = tqdm.tqdm(test_loader)
            for x, y, p in iterate:
                x = x.cuda()
                y = y.cuda()
                
                z, z_sample, x_pred = model(x)
                
                b, c, h, w = x.shape
                for i in range(b):
                    x_image = default_image_untransform(x[i])
                    Image.fromarray(x_image).save(
                        './target_%i_%i.png'%(epoch, i))
                    
                    c, h, w = x_pred[i].shape
                    xi = x_pred[i].view(c//3, 3, h, w)
                    xi = torch.argmax(xi, dim=0).permute(1,2,0)
                    xi = xi.cpu().numpy().astype(numpy.uint8)
                    Image.fromarray(xi).save(
                        './predicted_%i_%i.png'%(epoch, i))
                
                break

def encode_dataset(
    model_checkpoint,
    dataset_name,
    split,
    batch_size=4,
):
    dataset = FrameDataset(dataset_name, split=split, subset=None)
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=8)
    
    model = dvae.DVAE(
        hidden_channels=64,
        output_channels=(256*3),
        vocabulary_size=4096,
    ).cuda()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    
    cache = {'frames':{}, 'snaps':{}}
    with torch.no_grad():
        iterate = tqdm.tqdm(loader)
        for x, y, paths in iterate:
            x = x.cuda()
            z_sample = model(x, sample_only=True)
            z_sample = torch.argmax(z_sample, dim=1)
            
            for i, p in enumerate(paths):
                cache['frames'][p] = z_sample[i].cpu().numpy()
                cache['snaps'][p] = {}
                
                if 'color_x_' in p:
                    neg_snap_path = p.replace(
                        'color_x_', 'snap_neg_').replace(
                        '.png', '.npz'
                    )
                    pos_snap_path = neg_snap_path.replace(
                        'snap_neg_', 'snap_pos_')
                elif 'color_y_' in p:
                    # this is relying on the fact that the sequence always
                    # ends correct because I didn't write out the snap maps
                    # for the y images when generating frames (oopsie)
                    neg_snap_path = p.replace(
                        'color_y_', 'snap_neg_').replace(
                        '.png', '_0006.npz')
                    pos_snap_path = neg_snap_path.replace(
                        'snap_neg_', 'snap_pos_')
                #if (os.path.exists(neg_snap_path) and
                #    os.path.exists(pos_snap_path)
                #):
                neg_snap_map = numpy.load(neg_snap_path)['arr_0']
                pos_snap_map = numpy.load(pos_snap_path)['arr_0']
            
            
                # get wedge snaps
                for snap_id in range(9):
                    if snap_id <= 4:
                        snap_map = neg_snap_map
                    else:
                        snap_map = pos_snap_map
                    y, x = numpy.where(
                        (snap_map[:,:,0] == 1) &
                        (snap_map[:,:,1] == snap_id)
                    )
                    if len(y):
                        y_mean = numpy.sum(y)/len(y)
                        x_mean = numpy.sum(x)/len(x)
                        dy = y - y_mean
                        dx = x - x_mean
                        dd = dy**2 + dx**2
                        pixel_index = numpy.argmin(dd)
                        best_y = y[pixel_index]
                        best_x = x[pixel_index]
                        cache['snaps'][p][1, snap_id] = (best_y, best_x)
                
                # get slope_snaps
                for slope_id in 2,3:
                    for snap_id in range(3):
                        if snap_id == 0:
                            snap_map = pos_snap_map
                        else:
                            snap_map = neg_snap_map
                        y, x = numpy.where(
                            (snap_map[:,:,0] == slope_id) &
                            (snap_map[:,:,1] == snap_id)
                        )
                        if len(y):
                            y_mean = numpy.sum(y)/len(y)
                            x_mean = numpy.sum(x)/len(x)
                            dy = y - y_mean
                            dx = x - x_mean
                            dd = dy**2 + dx**2
                            pixel_index = numpy.argmin(dd)
                            best_y = y[pixel_index]
                            best_x = x[pixel_index]
                            cache['snaps'][p][slope_id, snap_id] = (
                                best_y, best_x)
    
    numpy.savez_compressed('./dvae_cache.npz', cache)

if __name__ == '__main__':
    #train()
    encode_dataset('./dvae_01/model_0032.pt', 'conditional_snap_two_frames', 'all')
