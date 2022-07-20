#!/usr/bin/env python
import random
import os

import numpy

import torch
from torch.nn import Module, Embedding, Linear, MultiheadAttention
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import tqdm

from ltron.config import Config
from ltron.compression import batch_deduplicate_tiled_seqs

from ltron_torch.train.optimizer import OptimizerConfig, adamw_optimizer
from ltron_torch.models.transformer import (
    TransformerConfig, Transformer)
from ltron_torch.models.compressed_transformer import (
    CompressedTransformerConfig, CompressedTransformer)
from ltron_torch.models.parameter import NoWeightDecayParameter
from ltron_torch.models.transformoencoder import Transformoencoder
from ltron_torch.models.slotoencoder import SlotoencoderConfig, Slotoencoder


# Build Dataset ================================================================

dataset_path = './int_pile'
from torch.utils.tensorboard import SummaryWriter
def make_scene(v, t, h, w):
    scene = numpy.zeros((h, w), dtype=numpy.long)
    
    assert t < h*w
    
    for i in range(t):
        token = random.randint(1, v-1)
        column_heights = get_column_heights(scene)
        while True:
            x = random.randint(0, w-1)
            column_height = column_heights[x]
            if column_height < h:
                scene[h-column_height-1, x] = token
                break
    
    return scene

def make_observation(scene, hidden_token=0):
    observation = numpy.copy(scene)
    h, w = scene.shape
    column_heights = get_column_heights(scene)
    for x in range(w):
        column_height = column_heights[x]
        for y in range(column_height - 1):
            observation[h-y-1, x] = hidden_token
    return observation

def get_column_heights(scene):
    empty = scene == 0
    anti_height = numpy.sum(empty, axis=0)
    return scene.shape[0] - anti_height

def print_scene(scene):
    max_len = len(str(numpy.max(scene)))
    print('+' + '-'*scene.shape[1]*(max_len+1) + '+')
    for row in scene:
        row_str = '|' + ''.join(
            ('%' + str(max_len+1) + 'i')%t
            if t != 0 else ' '*(max_len+1) for t in row) + '|'
        print(row_str)
    print('+' + '-'*scene.shape[1]*(max_len+1) + '+')

def expert_sequence(v, t, h, w):
    scene = make_scene(v, t, h, w)
    state = numpy.copy(scene)
    actions = []
    observations = []
    for t in range(t):
        observations.append(make_observation(state))
        column_heights = get_column_heights(state)
        nonempty_columns = numpy.nonzero(column_heights)[0]
        x = random.choice(nonempty_columns)
        y = h - column_heights[x]
        actions.append((x,y))
        state[y,x] = 0
    
    observations = numpy.stack(observations, axis=0)
    actions = numpy.array(actions)
    
    return scene, observations, actions

def make_dataset(n, v, t, h, w):
    dataset_path_v_txhxw = '%s_%i_%ix%ix%i'%(dataset_path, v, t, h, w)
    print('Making %s'%dataset_path_v_txhxw)
    if not os.path.exists(dataset_path_v_txhxw):
        os.makedirs(dataset_path_v_txhxw)
    
    for i in tqdm.tqdm(range(n)):
        s, o, a = expert_sequence(v, t, h, w)
        seq_path = os.path.join(dataset_path_v_txhxw, 'sequence_%06i.npz'%i)
        numpy.savez_compressed(seq_path, scene=s, observations=o, actions=a)


# Dataset ======================================================================

class IntPileSequenceDataset(Dataset):
    def __init__(self, train, v, t, h, w):
        self.dataset_path_v_txhxw = '%s_%i_%ix%ix%i'%(dataset_path, v, t, h, w)
        all_paths = os.listdir(self.dataset_path_v_txhxw)
        if train:
            self.paths = all_paths[:35000]
        else:
            self.paths = all_paths[35000:]
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        data = numpy.load(
            os.path.join(self.dataset_path_v_txhxw, self.paths[i]))
        scene = torch.LongTensor(data['scene'])
        observations = torch.LongTensor(data['observations'])
        actions = torch.LongTensor(data['actions'])
        
        return scene, observations, actions

def collate(data):
    scenes, observations, actions = zip(*data)
    scenes = torch.stack(scenes, dim=-1)
    observations = torch.stack(observations, dim=-1)
    actions = torch.stack(actions, dim=1)
    
    s, h, w, b = observations.shape
    scenes = scenes.view(h*w, b)
    observations = observations.view(s, h*w, b)
    
    return scenes, observations, actions

class IntPileDataset(Dataset):
    def __init__(self, train, v, t, h, w):
        self.t = t
        self.dataset_path_v_txhxw = '%s_%i_%ix%ix%i'%(dataset_path, v, t, h, w)
        all_paths = os.listdir(self.dataset_path_v_txhxw)
        if train:
            self.paths = all_paths[:35000]
        else:
            self.paths = all_paths[35000:]
    
    def __len__(self):
        return len(self.paths*self.t)
    
    def __getitem__(self, i):
        
        path_index = i // self.t
        observation_index = i % self.t
        
        data = numpy.load(os.path.join(
            self.dataset_path_v_txhxw, self.paths[path_index]))
        
        observation = torch.LongTensor(
            data['observations'][observation_index]).view(-1)
        
        return observation

def collate_single(data):
    observations = zip(*data)
    observations = torch.stack(data, dim=-1)
    
    return observations


# Compress Dataset =============================================================

def compress_dataset(epochs, v, t, h, w):
    train_dataset = IntPileDataset(True, v, t, h, w)
    train_loader = DataLoader(
        train_dataset, batch_size=16, collate_fn=collate_single, shuffle=True)
    test_dataset = IntPileDataset(False, v, t, h, w)
    test_loader = DataLoader(
        test_dataset, batch_size=16, collate_fn=collate_single)
    
    '''
    model = DualWeightCompressor(
        vocabulary=32,
        data_tokens=64,
        hidden_tokens=16,
        channels=256,
        num_heads=1,
    ).cuda()
    '''
    
    encoder_config = TransformerConfig(
        v = 32,
        t = 1,
        h = 8,
        w = 8,
        decoder_tokens = 16,
        decode_input = False,
        
        block_type = 'multi_pblock',
        num_blocks = 4,
        channels = 256,
        num_heads = 4,
        decoder_channels = 256,
    )
    decoder_config = TransformerConfig(
        #t = 16,
        t = 16,
        h = 1,
        w = 1,
        decoder_tokens = 8*8,
        #decoder_tokens = 0,
        decode_input = False,
        
        block_type = 'pblock',
        num_blocks = 4,
        channels = 256,
        num_heads = 4,
        decoder_channels = 32
    )
    model = Transformoencoder(encoder_config, decoder_config).cuda()
    
    optimizer_config = OptimizerConfig()
    optimizer = adamw_optimizer(model, optimizer_config)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    #class_weight = torch.ones(4096)
    class_weight = torch.ones(32)
    class_weight[0] = 0.1
    class_weight = class_weight.cuda()
    
    running_loss1 = 0.
    running_loss2 = 0.
    for epoch in range(1, epochs+1):
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for i, x in enumerate(iterate):
            t, b = x.shape
            
            x = x.cuda()
            y = x.cuda()
            
            #model.dropout.p = 1. - 0.99**i
            
            #x, w1, w2 = model(x.unsqueeze(0))
            x = model(x.unsqueeze(0))[-8*8:]
            hw, b, c = x.shape
            
            #b, c, h, w = x.shape
            
            #y = y.permute(1, 0).contiguous().view(b, h, w)
            
            #x = x[:,:,[0],:3]
            #y = y[:,[0],:3]
            '''
            fake_gt = torch.ones(b, 2, h, w) * 1000
            fake_gt[:,1] *= -1.
            fake_gt[0,0,0,0] *= -1.
            fake_gt[0,1,0,0] *= -1.
            fake_gt[1,0,0,1] *= -1.
            fake_gt[1,1,0,1] *= -1.
            fake_gt = fake_gt.cuda()
            '''
            loss1 = torch.nn.functional.cross_entropy(
                x.view(hw*b, c), y.view(hw*b), weight=class_weight)
            
            #loss2 = torch.nn.functional.mse_loss(
            #    w1, w2.permute(0,2,1)) * 0.1
            
            #yc = torch.LongTensor([0,1]).cuda()
            #loss2 = torch.nn.functional.cross_entropy(xc, yc)
            
            # WHY IS THIS LOSS NECESSARY
            loss = loss1 #+ loss2
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss1 = running_loss1 * 0.9 + float(loss1) * 0.1
            #running_loss2 = running_loss2 * 0.9 + float(loss2) * 0.1
            #iterate.set_description(
            #    'r: %.04f l: %.04f c:%.04f'%(running_loss, loss1, loss2))
            #iterate.set_description(
            #    'l1: %.04f, l2: %.04f'%(running_loss1, running_loss2))
            iterate.set_description('l1: %.04f'%running_loss1)
            
            if i % 64 == 0:
                #in0 = x_in[:,0].view(32,32).cpu().numpy()
                in0 = y[:,0].view(8,8).cpu().numpy()
                print_scene(in0)
                
                out0 = torch.argmax(x[:,0], dim=-1).view(8,8).cpu().numpy()
                print_scene(out0)
                
                in1 = y[:,1].view(8,8).cpu().numpy()
                print_scene(in1)
                
                out1 = torch.argmax(x[:,1], dim=-1).view(8,8).cpu().numpy()
                print_scene(out1)
                
# Model Utils ==================================================================

def get_num_model_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


# Train Reconstruction =========================================================

class ReconstructionConfig:
    epochs = 10
    batch_size = 16
    print_frequency = 128
    
    background_weight=0.05
    
    v = 32
    t = 32
    h = 8
    w = 8
    
    grid_t = 1
    grid_h = 1
    grid_w = 1
    
    model = 'time_then_space'
    channels = 512
    num_blocks = 4
    num_heads = 8
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key)
            setattr(self, key, value)

def delta_pack_observations(o, v, samples_per_image=2):
    t, hw, b = o.shape
    
    tokens = numpy.zeros((hw, b, 3), numpy.long)
    tokens[:,:,0] = o[0]
    tokens[:,:,2] = numpy.arange(hw).reshape(hw,1)
     
    for i in range(1, t):
        changes = numpy.where(o[i-1] != o[i])
        spatial_locations = numpy.random.randint(0, hw, (samples_per_image, b))
        
        #for j, k in zip(*changes):
        #    spatial_locations[
        
        import pdb
        pdb.set_trace()
        
        new_tokens[:,:,0] = o[i][changes[0], changes[1]]
        new_tokens[:,:,1] = i
        #new_tokens[:,:,2] = 

def test_delta_pack():
    train_dataset = IntPileSequenceDataset(
        train=True, v=4096, t=128, h=32, w=32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn = collate,
    )
    
    for s, o, a in train_loader:
        o = delta_pack_observations(o, 4096)

def train_encoder_reconstruction(config):
    print('making train dataset')
    train_dataset = IntPileSequenceDataset(
        train=True, v=config.v, t=config.t, h=config.h, w=config.w)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn = collate,
    )
    
    print('making test dataset')
    test_dataset = IntPileSequenceDataset(
        train=False, v=config.v, t=config.t, h=config.h, w=config.w)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
    )
    
    print('making model')
    if config.model in ('gpt', 'time_only', 'time_then_space',):
        model_config = TransformerConfig(
            t=config.t,
            v=config.v,
            h=config.h,
            w=config.w,
            grid_t=config.grid_t,
            grid_h=config.grid_h,
            grid_w=config.grid_w,
            decoder_tokens=config.h*config.w//(config.grid_h*config.grid_t),
            decode_input=False,
            
            attention_module='torch',
            
            block_type=config.model,
            
            num_blocks=config.num_blocks,
            channels=config.channels,
            residual_channels=None,
            num_heads=config.num_heads,
            decoder_channels=config.v*16,
        )
        model = Transformer(model_config).cuda()
    print('parameters: %i'%get_num_model_parameters(model))
    
    print('making optimizer')
    optimizer_config = OptimizerConfig()
    optimizer = adamw_optimizer(model, optimizer_config)
    
    class_weight = torch.ones(config.v)
    class_weight[0] = config.background_weight
    class_weight = class_weight.cuda()
    
    running_loss = 0.0
    running_accuracy = 0.0
    for epoch in range(1, config.epochs+1):
        print('epoch: %i/%i'%(epoch, config.epochs))
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for i, (s, o, a) in enumerate(iterate):
            s = s.cuda()
            o = o.cuda()
            t, hw, b = o.shape
            
            x = model(o)
            thw, b, c = x.shape
            #x = x.view(-1, model_config.hh * model_config.ww, b, c)[0]
            
            # oof...
            grid_cells = (
                model_config.grid_h * model_config.grid_w)
            cc = c//grid_cells
            x = x.view(
                model_config.hh,
                model_config.ww,
                b,
                model_config.grid_h,
                model_config.grid_w,
                cc,
            )
            x = x.permute(2,0,3,1,4,5).contiguous()
            x = x.view(b, model_config.h, model_config.w, cc)
            
            x = x.permute(1,2,0,3).contiguous().view(
                model_config.h*model_config.w, b, cc)
            
            loss = torch.nn.functional.cross_entropy(
                x.view(-1,cc), s.view(-1), weight=class_weight)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            estimate = torch.argmax(x, dim=-1)
            correct = estimate == s
            accuracy = torch.sum(correct * (s != 0)).float() / torch.sum(s != 0)
            
            running_loss = running_loss * 0.9 + float(loss)*0.1
            running_accuracy = running_accuracy * 0.9 + float(accuracy)*0.1
            iterate.set_description(
                'l: %.04f a: %.04f'%(running_loss, running_accuracy))
            
            if i%config.print_frequency == 0:
                print_scene(estimate[:,0].view(
                    model_config.h,model_config.w).cpu().numpy())
                print_scene(s[:,0].view(
                    model_config.h,model_config.w).cpu().numpy())


class CompressedConfig:
    epochs = 10
    batch_size = 16
    print_frequency = 128
    
    background_weight=0.05
    
    v = 32
    t = 32
    h = 8
    w = 8
    
    channels = 512
    num_blocks = 4
    num_heads = 8
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key)
            setattr(self, key, value)


def train_compressed(config):
    print('making log')
    log = SummaryWriter()
    
    print('making train dataset')
    train_dataset = IntPileSequenceDataset(
        train=True, v=config.v, t=config.t, h=config.h, w=config.w)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn = collate,
    )
    
    print('making test dataset')
    test_dataset = IntPileSequenceDataset(
        train=False, v=config.v, t=config.t, h=config.h, w=config.w)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn = collate
    )
    
    print('making model')
    model_config = CompressedTransformerConfig(
        t=config.t+1,
        h=config.h,
        w=config.w,
        tile_h=1,
        tile_w=1,
        decoder_tokens=0,
        decode_input=True,
        
        relative_positional_encoding=True,
        factored_positional_encoding=True,
        learned_positional_encoding=True,
        
        num_blocks=config.num_blocks,
        channels=config.channels,
        residual_channels=None,
        num_heads=config.num_heads,
        decoder_channels=config.v,
    )
    model = CompressedTransformer(model_config).cuda()
    embedding = Embedding(config.v, 3).cuda()
    decoder_embedding = Embedding(1,3).cuda()
    print('parameters: %i'%get_num_model_parameters(model))
    
    print('making optimizer')
    optimizer_config = OptimizerConfig()
    optimizer = adamw_optimizer(model, optimizer_config)
    
    class_weight = torch.ones(config.v)
    class_weight[0] = config.background_weight
    class_weight = class_weight.cuda()
    
    step = 0
    running_loss = 0.0
    running_accuracy = 0.0
    for epoch in range(1, config.epochs+1):
        print('epoch: %i/%i'%(epoch, config.epochs))
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for i, (s, o, a) in enumerate(iterate):
            s = s.cuda()
            #o = o.cuda()
            t, hw, b = o.shape
            
            o = o.view(t, config.h, config.w, b)
            o = o.permute(3,0,1,2).cpu().detach().numpy()
            co, j, m = batch_deduplicate_tiled_seqs(o, 1, 1)
            co = embedding(torch.LongTensor(co).cuda())
            jh = j[:,:,1] // config.w
            jw = j[:,:,1] % config.w
            j = torch.LongTensor(
                numpy.stack((j[:,:,0], jh, jw), axis=-1)).cuda()
            
            nd = config.h * config.w
            decoder_tokens = torch.zeros((nd, b), dtype=torch.long).cuda()
            decoder_tokens = decoder_embedding(decoder_tokens)
            decoder_tokens = decoder_tokens.view(nd, b, 1, 1, 3)
            co = torch.cat((co, decoder_tokens), dim=0)
            decoder_j = torch.zeros((config.h,config.w,b,3), dtype=torch.long)
            decoder_j[:,:,:,0] = config.t
            decoder_j[:,:,:,1] = torch.arange(config.h).view(config.h,1,1)
            decoder_j[:,:,:,2] = torch.arange(config.w).view(1,config.w,1)
            decoder_j = decoder_j.view(nd, b, 3).cuda()
            j = torch.cat((j, decoder_j), dim=0)
            
            x = model(co, j, m)
            thw, b, c = x.shape
            
            x = x[-nd:]
            
            loss = torch.nn.functional.cross_entropy(
                x.view(-1,c), s.view(-1), weight=class_weight)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            estimate = torch.argmax(x, dim=-1)
            correct = estimate == s
            accuracy = torch.sum(correct * (s != 0)).float() / torch.sum(s != 0)
            
            running_loss = running_loss * 0.9 + float(loss)*0.1
            running_accuracy = running_accuracy * 0.9 + float(accuracy)*0.1
            iterate.set_description(
                'l: %.04f a: %.04f'%(running_loss, running_accuracy))
            
            log.add_scalar('Loss/train', float(loss), step)
            log.add_scalar('Accuracy/train', float(accuracy), step)
            step += 1
            
            '''
            if i%config.print_frequency == 0:
                print_scene(estimate[:,0].view(
                    model_config.h,model_config.w).cpu().numpy())
                print_scene(s[:,0].view(
                    model_config.h,model_config.w).cpu().numpy())
            '''

if __name__ == '__main__':
    #scene = make_scene(100, 32, 8, 8)
    #observation = make_observation(scene)
    #print_scene(scene)
    #print_scene(observation)
    
    #scene, observations, actions = expert_sequence(32, 32, 8, 8)
    #print_scene(scene)
    #for observation, action in zip(observations, actions):
    #    print_scene(observation)
    #    print(action)
    
    #make_dataset(40000, 32, 32, 8, 8)
    #make_dataset(40000, 4096, 128, 32, 32)
    
    #make_dataset(40000, 4096, 128, 16, 16)
    
    #config = ReconstructionConfig(
    #    epochs = 300,
    #    model='time_then_space',
    #    grid_t = 4,
    #    grid_h = 4,
    #    grid_w = 4,
    #    v=4096,
    #    t=128,
    #    h=32,
    #    w=32,
    #    batch_size=4,
    #)
    #train_encoder_reconstruction(config)
    
    #test_delta_pack()
    
    #compress_dataset(10, 32, 32, 8, 8)
    
    config = CompressedConfig(
        epochs = 10,
        v=4096,
        t=128,
        h=16,
        w=16,
        #v=32,
        #t=32,
        #h=8,
        #w=8,
        
        channels = 256,
        num_heads = 4,
        batch_size=2,
        num_blocks=6,
    )
        
    train_compressed(config)
