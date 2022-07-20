#!/usr/bin/env python
import os

import torch
from torch.utils.data import Dataset, DataLoader

import numpy

from PIL import Image

import tqdm

from ltron.hierarchy import len_hierarchy
from ltron.dataset.paths import get_dataset_paths
from ltron.visualization.drawing import block_upscale_image

from ltron_torch.models.positional_encoding import raw_positional_encoding

class CachedDataset(Dataset):
    def __init__(self, data, dataset_name, split, subset=None):
        paths = get_dataset_paths(dataset_name, split, subset=subset)
        self.frame_order = []
        for i in range(len_hierarchy(paths)):
            for j in range(7):
                self.frame_order.append(paths['color_x'][i].replace(
                    '_0000.png', '_%04i.png'%j))
        
        self.data = data
        #self.frame_order = sorted(list(self.data['frames'].keys()))
    
    def __len__(self):
        return len(self.frame_order)
    
    def __getitem__(self, index):
        frame_path = self.frame_order[index]
        x = self.data['frames'][frame_path]
        y = numpy.array([
            self.data['snaps'][frame_path].get((2,1),(0,0)),
            self.data['snaps'][frame_path].get((2,2),(0,0)),
            self.data['snaps'][frame_path].get((3,1),(0,0)),
            self.data['snaps'][frame_path].get((3,2),(0,0)),
        ])
        
        return x, y, frame_path

class Model(torch.nn.Module):
    def __init__(
        self,
        channels=256,
        transformer_layers=6,
        transformer_heads=4,
        transformer_dropout=0.5,
        embedding_dropout=0.1,
    ):
        super(Model, self).__init__()
        self.transformer_heads = transformer_heads
        self.transformer_dropout = transformer_dropout
        
        self.register_buffer(
            'sequence_encoding',
            raw_positional_encoding(channels, 32**2+4).unsqueeze(1),
        )
        
        self.embedding = torch.nn.Embedding(4192, channels)
        self.out_embedding = torch.nn.Embedding(4, channels)
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            self.transformer_heads,
            channels,
            self.transformer_dropout,
        )
        self.transformer = torch.nn.TransformerEncoder(
            transformer_layer, transformer_layers)
        
        self.out_linear = torch.nn.Linear(channels, 64+64)
    
    def forward(self, x):
        b,h,w = x.shape
        x = x.view(b, h*w).permute(1,0)
        x = self.embedding(x)
        out = torch.arange(4).view(4,1).expand(4,b).to(x.device)
        out = self.out_embedding(out)
        x_out = torch.cat((out, x), dim=0)
        x_out = x_out + self.sequence_encoding
        
        x_out = self.transformer(x_out)
        
        out = self.out_linear(x_out[:4])
        
        return out

def train(
    num_epochs=50,
):
    print('loading dataset')
    cached_data = numpy.load(
        './dvae_cache.npz', allow_pickle=True)['arr_0'].item()
    train_dataset = CachedDataset(
        cached_data,
        'conditional_snap_two_frames',
        'train',
        subset=None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
    )
    
    test_dataset = CachedDataset(
        cached_data,
        'conditional_snap_two_frames',
        'test',
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
    )
    
    print('making model')
    model = Model().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for epoch in range(1, num_epochs+1):
        print('epoch: %i'%epoch)
        print('train')
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for x, y, paths in iterate:
            x = x.cuda()
            y = y.cuda().permute(1,0,2)
            s, b, _ = y.shape
            y = y.reshape(s*b, 2)
            
            x = model(x)
            s, b, c = x.shape
            x = x.view(s*b, 64, 2)
            
            loss = torch.nn.functional.cross_entropy(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            iterate.set_description('loss: %.04f'%float(loss))
        
        print('test')
        model.eval()
        with torch.no_grad():
            iterate = tqdm.tqdm(test_loader)
            for x, y, paths in iterate:
                x = x.cuda()
                y = y.cuda().permute(1,0,2)
                #s, b, _ = y.shape
                #y = y.reshape(x*b, 2)
                
                pred = model(x)
                s, b, c = pred.shape
                pred = pred.view(s, b, 64, 2)
                pred = torch.argmax(pred, dim=2).cpu().numpy()
                
                for i in range(b):
                    p = pred[:,i]
                    pred_drawing = numpy.zeros((64, 64, 1))
                    pred_drawing[p[:,0],p[:,1]] = 1.
                    pred_drawing = block_upscale_image(pred_drawing, 256, 256)
                    
                    path = paths[i]
                    image = numpy.array(Image.open(path))
                    image = (
                        image * (1. - pred_drawing) +
                        numpy.array([[[255,0,0]]]) * pred_drawing)
                    image = image.astype(numpy.uint8)
                    image_path = os.path.join(
                        '.', 'epoch_%i_'%epoch + os.path.basename(path))
                    Image.fromarray(image).save(image_path)
                break
        
        torch.save(model.state_dict(), 'transformer_locate_model_%04i.pt'%epoch)

if __name__ == '__main__':
    train()
