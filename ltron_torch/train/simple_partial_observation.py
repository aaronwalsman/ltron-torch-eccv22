#!/usr/bin/env python
import random
import os

import numpy

import torch
from torch.nn import Module, Embedding, Linear, MultiheadAttention
from torch.utils.data import Dataset, DataLoader

import tqdm

from ltron_torch.models.transformer import (
    TransformerConfig, TokenMapSequenceEncoder)

from ltron_torch.train.optimizer import OptimizerConfig, adamw_optimizer

from ltron_torch.models.parameter import NoWeightDecayParameter
from ltron_torch.models.transformoencoder import Transformoencoder
from ltron_torch.models.slotoencoder import SlotoencoderConfig, Slotoencoder

# Build Dataset ================================================================

dataset_path = './simple_partial_observation'

def make_scene(h, w, vocabulary, tokens):
    scene = numpy.zeros((h, w), dtype=numpy.long)
    
    assert tokens < h*w
    
    for i in range(tokens):
        token = random.randint(1, vocabulary-1)
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

def expert_sequence(h, w, vocabulary, tokens):
    scene = make_scene(h, w, vocabulary, tokens)
    state = numpy.copy(scene)
    actions = []
    observations = []
    for t in range(tokens):
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

def make_dataset(n, h, w, vocabulary, tokens):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    for i in tqdm.tqdm(range(n)):
        s, o, a = expert_sequence(h, w, vocabulary, tokens)
        seq_path = os.path.join(dataset_path, 'sequence_%06i.npz'%i)
        numpy.savez_compressed(seq_path, scene=s, observations=o, actions=a)


# Dataset ======================================================================

class SimplePartialObservationDataset(Dataset):
    def __init__(self, train):
        all_paths = os.listdir(dataset_path)
        if train:
            self.paths = all_paths[:35000]
        else:
            self.paths = all_paths[35000:]
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        data = numpy.load(os.path.join(dataset_path, self.paths[i]))
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

class SingleFrameSimplePartialObservationDataset(Dataset):
    def __init__(self, train, sequence_length):
        self.sequence_length = sequence_length
        all_paths = os.listdir(dataset_path)
        if train:
            self.paths = all_paths[:35000]
        else:
            self.paths = all_paths[35000:]
    
    def __len__(self):
        return len(self.paths*self.sequence_length)
    
    def __getitem__(self, i):
        
        path_index = i // self.sequence_length
        observation_index = i % self.sequence_length
        
        data = numpy.load(os.path.join(dataset_path, self.paths[path_index]))
        
        observation = torch.LongTensor(
            data['observations'][observation_index]).view(-1)
        
        return observation

def collate_single(data):
    observations = zip(*data)
    observations = torch.stack(data, dim=-1)
    
    return observations

# Compress Dataset =============================================================

class DualWeightCompressor(Module):
    def __init__(
        self, vocabulary, data_tokens, hidden_tokens, channels, num_heads
    ):
        super(DualWeightCompressor, self).__init__()
        self.attention1 = torch.nn.MultiheadAttention(channels, num_heads)
        self.attention2 = torch.nn.MultiheadAttention(channels, num_heads)
        
        self.data_embedding = Embedding(vocabulary, channels)
        self.hidden_embedding = Embedding(hidden_tokens, channels)
        #self.hidden_predictor = Linear(channels, hidden_tokens)
        
        pe1 = torch.zeros(data_tokens, 1, channels)
        self.positional_encoding1 = NoWeightDecayParameter(pe1)
        
        pe2 = torch.zeros(data_tokens, 1, channels)
        self.positional_encoding2 = NoWeightDecayParameter(pe2)
        
        self.out_linear = Linear(channels, vocabulary)
    
    def forward(self, x):
        x = self.data_embedding(x)
        t, hw, b, c = x.shape
        x = x.view(t*hw, b, c)
        x = x + self.positional_encoding1
        #h = self.hidden_predictor(x)
        #h = torch.softmax(x, dim=1)
        t, c = self.hidden_embedding.weight.shape
        s = self.hidden_embedding.weight.view(t, 1, c).expand(t, b, c)
        x, w1 = self.attention1(s, x, x)
        p = self.positional_encoding2
        t, _, c = p.shape
        p = p.expand(t, b, c)
        x, w2 = self.attention2(p, x, x)
        x = self.out_linear(x)
        
        return x, w1, w2

def compress_dataset(epochs=10):
    train_dataset = SingleFrameSimplePartialObservationDataset(
        train=True, sequence_length=32)
    train_loader = DataLoader(
        train_dataset, batch_size=16, collate_fn=collate_single, shuffle=True)
    test_dataset = SingleFrameSimplePartialObservationDataset(
        train=False, sequence_length=32)
    test_loader = DataLoader(
        test_dataset, batch_size=16, collate_fn=collate_single)
    
    '''
    sloto_config = SlotoencoderConfig(
        num_data_tokens=64,
        num_slots=1,
        vocabulary=2,
        output_channels=2,
        tile_shape=(8,8),
        num_decoder_blocks=1,
    )
    model = Slotoencoder(sloto_config).cuda()
    '''
    
    '''
    encoder_config = TransformerConfig(
        vocabulary=2,
        map_height=8,
        map_width=8,
        decoder_tokens=4,
        decode_input=False,
        
        block_type='gpt',
        num_blocks=4,
        channels=256,
        num_heads=1,
        learned_positional_encoding=True,
        decoder_channels=256,
        
        #attention_dropout=0,
        #residual_dropout=0,
    )
    
    decoder_config = TransformerConfig(
        decoder_tokens=8*8,
        sequence_length=4,
        map_height=1,
        map_width=1,
        decode_input=False,
        
        block_type='gpt',
        num_blocks=4,
        channels=256,
        num_heads=1,
        learned_positional_encoding=True,
        decoder_channels=2,
    )
    
    model = Transformoencoder(encoder_config, decoder_config).cuda()
    '''
    
    model = DualWeightCompressor(
        vocabulary=32,
        data_tokens=64,
        hidden_tokens=16,
        channels=256,
        num_heads=1,
    ).cuda()
    
    optimizer_config = OptimizerConfig()
    optimizer = adamw_optimizer(model, optimizer_config)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    #class_weight = torch.ones(4096)
    class_weight = torch.ones(32)
    class_weight[0] = 0.1
    class_weight = class_weight.cuda()
    
    #torch.autograd.set_detect_anomaly(True)
    running_loss1 = 0.
    running_loss2 = 0.
    for epoch in range(1, epochs+1):
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for i, x in enumerate(iterate):
            t, b = x.shape
            '''
            x = torch.zeros(64, b, dtype=torch.long)
            hw = torch.randint(64, (b,))
            #hw = torch.randint(3, (b,))
            #hw = torch.zeros(b).long()
            #x[0,0] = 1
            #x[1,1] = 1
            x[hw,range(b)] = 1
            hw = torch.randint(64, (b,))
            x[hw,range(b)] = 1
            hw = torch.randint(64, (b,))
            x[hw, range(b)] = 1
            '''
            x_in = x
            
            x = x.cuda()
            
            y = x.cuda()
            
            pause = i > 500
            #pause = False
            #x, xc = model(x, pause=pause)
            x, w1, w2 = model(x.unsqueeze(0))
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
            
            loss2 = torch.nn.functional.mse_loss(
                w1, w2.permute(0,2,1)) * 0.1
            
            #yc = torch.LongTensor([0,1]).cuda()
            #loss2 = torch.nn.functional.cross_entropy(xc, yc)
            
            # WHY IS THIS LOSS NECESSARY
            loss = loss1 + loss2
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss1 = running_loss1 * 0.9 + float(loss1) * 0.1
            running_loss2 = running_loss2 * 0.9 + float(loss2) * 0.1
            #iterate.set_description(
            #    'r: %.04f l: %.04f c:%.04f'%(running_loss, loss1, loss2))
            iterate.set_description(
                'l1: %.04f, l2: %.04f'%(running_loss1, running_loss2))
            
            if i % 64 == 0:
                #in0 = x_in[:,0].view(32,32).cpu().numpy()
                in0 = x_in[:,0].view(8,8).cpu().numpy()
                print_scene(in0)
                
                out0 = torch.argmax(x[:,0], dim=-1).view(8,8).cpu().numpy()
                print_scene(out0)
                
                in1 = x_in[:,1].view(8,8).cpu().numpy()
                print_scene(in1)
                
                out1 = torch.argmax(x[:,1], dim=-1).view(8,8).cpu().numpy()
                print_scene(out1)
                

def lossless_compression():
    train_dataset = SingleFrameSimplePartialObservationDataset(train=True)
    train_loader = DataLoader(
        train_dataset, batch_size=16, collate_fn=collate_single)
    
    iterate = tqdm.tqdm(train_loader)
    for i, x in enumerate(iterate):
        t, b = x.shape
        import pdb
        pdb.set_trace()

# Model Utils ==================================================================

def get_num_model_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


# Encoder Reconstruction =======================================================

class AttentionModel(torch.nn.Module):
    def __init__(self,
        channels,
        residual_channels,
        num_heads,
        num_layers,
        tokens_per_image
    ):
        super(AttentionModel, self).__init__()
        if residual_channels is None:
            residual_channels = channels*4
        self.embedding = torch.nn.Embedding(4096, channels)
        
        self.layers = torch.nn.Sequential(
            *[AttentionLayer(channels, residual_channels, num_heads)
              for _ in range(num_layers)])
        
        self.channels = channels
        self.tokens_per_image = tokens_per_image
    
    def forward(self, x):
        x = self.embedding(x)
        t, hw, b, c = x.shape
        x = x.permute(1,0,2,3).contiguous().view(hw, t*b, c)
        d = torch.randn((self.tokens_per_image, t*b, c)).cuda()
        
        xd = (x,d)
        xd = self.layers(xd)
        
        import pdb
        pdb.set_trace()
    
    def configure_optimizer(self, config):
        return torch.optim.AdamW(self.parameters(), lr=3e-4)

class AttentionLayer(torch.nn.Module):
    def __init__(self, channels, residual_channels, num_heads):
        super(AttentionLayer, self).__init__()
        self.ln1 = torch.nn.LayerNorm(channels)
        self.data_att = torch.nn.MultiheadAttention(channels, num_heads)
        
        self.ln2 = torch.nn.LayerNorm(channels)
        self.decode_att = torch.nn.MultiheadAttention(channels, num_heads)
        
        self.ln3 = torch.nn.LayerNorm(channels)
        self.linear1 = torch.nn.Linear(channels, residual_channels)
        self.nonlinear = torch.nn.ReLU(channels)
        self.linear2 = torch.nn.Linear(residual_channels, channels)
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, xd):
        x, d = xd
        r1 = self.ln1(d)
        r1 = self.data_att(r1, x, x)[0]
        d = d + r1
        
        r2 = self.ln2(d)
        r2 = self.decode_att(r2, r2, r2)[0]
        d = d + r2
        
        r3 = self.ln3(d)
        r3 = self.linear1(r3)
        r3 = self.nonlinear(r3)
        r3 = self.linear2(r3)
        r3 = self.dropout(r3)
        d = d + r3
        
        return x, d

class TestModel(torch.nn.Module):
    def __init__(self, channels, num_layers):
        super(TestModel, self).__init__()
        self.embedding = torch.nn.Embedding(4096, channels)
        self.layers = torch.nn.Sequential(
            *[TestLayer(channels) for _ in range(num_layers)])
        self.out = torch.nn.Linear(channels, 4096)
    
    def forward(self, x):
        print('in:', x.shape)
        x = self.embedding(x)
        x = torch.cat((torch.zeros(1, 1024, 8, 128).cuda(), x), dim=0)
        t, hw, b, c = x.shape
        x = x.view(t*hw, b, c)
        print('embed:', x.shape)
        x = self.layers(x)
        print('layers:', x.shape)
        x = x.view(t, hw, b, c)
        x = self.out(x[0])
        print('out:', x.shape)
        return x
    
    def configure_optimizer(self, config):
        return torch.optim.AdamW(self.parameters(), lr=3e-4)

class TestLayer(torch.nn.Module):
    def __init__(self, channels):
        super(TestLayer, self).__init__()
        self.attention_residual = torch.nn.Sequential(
            torch.nn.LayerNorm(channels)
        )
        self.projection_residual = torch.nn.Sequential(
            torch.nn.LayerNorm(channels),
            torch.nn.Linear(channels, channels),
            torch.nn.ReLU(),
            torch.nn.Linear(channels, channels),
            torch.nn.Dropout(0.1),
        )
    
    def forward(self, x):
        thw, b, c = x.shape
        x = x.view(thw*b, c)
        x = x + self.attention_residual(x)
        x = x + self.projection_residual(x)
        x = x.view(thw, b, c)
        return x

def train_encoder_reconstruction_cheating(epochs=10):
    print('making train dataset')
    train_dataset = SimplePartialObservationDataset(train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn = collate,
    )
    
    print('making test dataset')
    test_dataset = SimplePartialObservationDataset(train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
    )
    
    print('making model')
    model_config = TransformerConfig(
        sequence_length=33,
        vocabulary=33,
        map_height=8,
        map_width=8,
        decoder_tokens=0, #8*8,
        decode_input=True, #False,
        
        attention_module='torch',
        
        block_type='time_only',
        
        num_blocks=4,
        channels=512,
        residual_channels=None,
        num_heads=8,
        decoder_channels=32,
        
        learned_positional_encoding=False,
        
    )
    model = TokenMapSequenceEncoder(model_config).cuda()
    #model = TestModel(channels=128, num_layers=2).cuda()
    #model = AttentionModel(
    #    channels=128,
    #    residual_channels=None,
    #    num_heads=4,
    #    num_layers=4,
    #    tokens_per_image=8,
    #).cuda()
    print('parameters: %i'%get_num_model_parameters(model))
    
    print('making optimizer')
    optimizer_config = OptimizerConfig()
    optimizer = adamw_optimizer(model, optimizer_config)
    
    class_weight = torch.ones(32)
    class_weight[0] = 0.05
    class_weight = class_weight.cuda()
    
    running_loss = 0.0
    running_accuracy = 0.0
    for epoch in range(1,epochs+1):
        print('epoch: %i'%epoch)
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for i, (s, o, a) in enumerate(iterate):
            s = s.cuda()
            o = o.cuda()
            
            t, hw, b = o.shape
            
            decoder_tokens = torch.ones((1, 64, b), dtype=torch.long).cuda()*32
            o = torch.cat((o, decoder_tokens), dim=0)
            
            x = model(o)
            
            #t, b, c = x.shape
            thw, b, c = x.shape
            x = x.view(33,64,b,c)[-1]
            
            target = s.view(-1)
            #target = torch.zeros(s.shape, dtype=torch.long).view(-1).cuda()
            loss = torch.nn.functional.cross_entropy(
                x.view(-1,c), target, weight=class_weight)
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
            
            if i%128 == 0:
                print_scene(estimate[:,0].view(8,8).cpu().numpy())
                print_scene(s[:,0].view(8,8).cpu().numpy())

def train_encoder_reconstruction(epochs=10):
    print('making train dataset')
    train_dataset = SimplePartialObservationDataset(train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn = collate,
    )
    
    print('making test dataset')
    test_dataset = SimplePartialObservationDataset(train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        collate_fn = collate,
    )
    
    print('making model')
    model_config = TransformerConfig(
        sequence_length=33,
        vocabulary=33,
        map_height=8,
        map_width=8,
        decoder_tokens=0, #8*8,
        decode_input=True, #False,
        
        attention_module='torch',
        
        block_type='time_then_space',
        
        num_blocks=4,
        channels=512,
        residual_channels=None,
        num_heads=8,
        decoder_channels=32,
        
        learned_positional_encoding=False,
        
    )
    model = TokenMapSequenceEncoder(model_config).cuda()
    #model = TestModel(channels=128, num_layers=2).cuda()
    #model = AttentionModel(
    #    channels=128,
    #    residual_channels=None,
    #    num_heads=4,
    #    num_layers=4,
    #    tokens_per_image=8,
    #).cuda()
    print('parameters: %i'%get_num_model_parameters(model))
    
    print('making optimizer')
    optimizer_config = OptimizerConfig()
    optimizer = adamw_optimizer(model, optimizer_config)
    
    class_weight = torch.ones(32)
    class_weight[0] = 0.05
    class_weight = class_weight.cuda()
    
    running_loss = 0.0
    running_accuracy = 0.0
    for epoch in range(1,epochs+1):
        print('epoch: %i'%epoch)
        model.train()
        iterate = tqdm.tqdm(train_loader)
        for i, (s, o, a) in enumerate(iterate):
            s = s.cuda()
            o = o.cuda()
            
            t, hw, b = o.shape
            
            decoder_tokens = torch.ones((1, 64, b), dtype=torch.long).cuda()*32
            o = torch.cat((o, decoder_tokens), dim=0)
            
            x = model(o)
            
            #t, b, c = x.shape
            thw, b, c = x.shape
            x = x.view(33,64,b,c)[-1]
            
            target = s.view(-1)
            #target = torch.zeros(s.shape, dtype=torch.long).view(-1).cuda()
            loss = torch.nn.functional.cross_entropy(
                x.view(-1,c), target, weight=class_weight)
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
            
            if i%128 == 0:
                print_scene(estimate[:,0].view(8,8).cpu().numpy())
                print_scene(s[:,0].view(8,8).cpu().numpy())


if __name__ == '__main__':
    #scene = make_scene(32, 32, 100, 128)
    #observation = make_observation(scene)
    #print_scene(scene)
    #print_scene(observation)
    
    #scene, observations, actions = expert_sequence(16, 16, 10, 32)
    #print_scene(scene)
    #for observation, action in zip(observations, actions):
    #    print_scene(observation)
    #    print(action)
    
    #make_dataset(40000, 8, 8, 32, 32)
    
    train_encoder_reconstruction()
    
    #compress_dataset()

    #lossless_compression()
