#!/usr/bin/env python
import random
import time
import os

import numpy

import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from gym import Env
from gym.spaces import Discrete, MultiDiscrete
from gym.vector.async_vector_env import AsyncVectorEnv

from ltron.config import Config
from ltron.dataset.paths import get_dataset_info
from ltron.gym.ltron_env import async_ltron
from ltron.gym.reassembly_env import reassembly_env
from ltron.gym.rollout_storage import RolloutStorage
from ltron.compression import batch_deduplicate_tiled_seqs

from ltron_torch.gym_tensor import gym_space_to_tensors
from ltron_torch.train.optimizer import OptimizerConfig, adamw_optimizer
from ltron_torch.models.compressed_transformer import (
    CompressedTransformer, CompressedTransformerConfig)
from ltron_torch.models.padding import linearize_padded_seq


# Test Environments ============================================================

class NextEnv(Env):
    def __init__(self, count_max=32):
        self.count_max = count_max
        self.observation_space = MultiDiscrete((count_max, count_max))
        self.action_space = Discrete(count_max)
    
    def reset(self):
        self.t = 0
        self.index = random.randint(0, self.count_max-1)
        return (self.t, self.index)
    
    def step(self, action):
        if action == self.index+1:
            reward = 1
        else:
            reward = 0
        self.t += 1
        self.index += 1
        return (self.t, self.index), reward, self.index == self.count_max, {}

class CountEnv(Env):
    def __init__(self, episode_len=32):
        self.episode_len = episode_len
        self.observation_space = MultiDiscrete((episode_len, 2))
        self.action_space = Discrete(episode_len)
    
    def reset(self):
        self.count = 0
        self.t = 0
        obs = random.random() < 0.05
        self.count += obs
        return (self.t, obs)
    
    def step(self, action):
        reward = action == self.count
        obs = random.random() < 0.05
        self.count += obs
        self.t += 1
        return (self.t, obs), reward, self.t == self.episode_len, {}
    
class MaxEnv(Env):
    def __init__(self, num_values=32, episode_len=32):
        self.num_values = num_values
        self.episode_len = episode_len
        self.observation_space = MultiDiscrete((episode_len, num_values))
        self.action_space = Discrete(num_values)
    
    def reset(self):
        obs = random.randint(0, self.num_values-1)
        self.max_value = obs
        self.t = 0
        return (self.t, obs)
    
    def step(self, action):
        reward = action == self.max_value
        obs = random.randint(0, self.num_values-1)
        self.max_value = max(self.max_value, obs)
        self.t += 1
        return (self.t, obs), reward, self.t == self.episode_len, {}

class VariableMaxEnv(Env):
    def __init__(self, num_values=32, p_end=0.05, max_len=32):
        self.num_values = num_values
        self.p_end = p_end
        self.observation_space = MultiDiscrete((10000, num_values))
        self.action_space = Discrete(num_values)
        self.max_len = max_len
    
    def reset(self):
        obs = random.randint(0, self.num_values-1)
        self.max_value = obs
        self.t = 0
        return (self.t, obs)
    
    def step(self, action):
        reward = action == self.max_value
        obs = random.randint(0, self.num_values-1)
        self.max_value = max(self.max_value, obs)
        self.t += 1
        terminal = random.random() < self.p_end or self.t >= self.max_len
        return (self.t, obs), reward, terminal, {}


# Train Config =================================================================

class TrainConfig(Config):
    epochs=10
    training_passes_per_epoch=8
    batch_size=16
    num_envs=16
    rollout_steps_per_epoch=2048*4
    
    env = 'max'
    
    test_frequency=None
    checkpoint_frequency=1
    
    def set_dependents(self):
        self.batch_rollout_steps_per_epoch = (
            self.rollout_steps_per_epoch // self.num_envs
        )


# Training Scripts =============================================================

def train_compressed_transformer(train_config):
    
    print('='*80)
    print('Setup')
    print('-'*80)
    print('Log')
    log = SummaryWriter()
    clock = [0]
    
    print('-'*80)
    print('Building Model')
    model_config = CompressedTransformerConfig(
        data_shape=(32,),
        tile_h=1,
        tile_w=1,
        
        num_blocks=4,
        channels=256,
        num_heads=4,
        
        input_mode='token',
        input_token_vocab=32,
        
        decode_input=True,
        decoder_channels=32+1,
        
        content_dropout = 0.,
        embedding_dropout = 0.,
        attention_dropout = 0.,
        residual_dropout= 0.,
    )
    model = CompressedTransformer(model_config).cuda()
    
    print('-'*80)
    print('Building Optimizer')
    optimizer_config = OptimizerConfig(learning_rate=3e-5)
    optimizer = adamw_optimizer(model, optimizer_config)
    
    print('-'*80)
    print('Building Envs')
    if train_config.env == 'next':
        constructors = [NextEnv for i in range(train_config.num_envs)]
    elif train_config.env == 'count':
        constructors = [CountEnv for i in range(train_config.num_envs)]
    elif train_config.env == 'max':
        constructors = [MaxEnv for i in range(train_config.num_envs)]
    elif train_config.env == 'variable_max':
        constructors = [VariableMaxEnv for i in range(train_config.num_envs)]
    envs = AsyncVectorEnv(constructors, context='spawn')
    
    for epoch in range(1, train_config.epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        
        episodes = rollout_epoch(train_config, epoch, envs, model, log, clock)
        train_epoch(train_config, epoch, model, optimizer, episodes, log, clock)
        save_checkpoint(train_config, epoch, model, optimizer, log, clock)
        
        epoch_end = time.time()
        print('Elapsed: %.02f seconds'%(epoch_end-epoch_start))

def rollout_epoch(train_config, epoch, env, model, log, clock):
    print('-'*80)
    print('Rolling Out Episodes')
    
    # initialize storage for observations, actions and rewards
    observation_storage = RolloutStorage(train_config.num_envs)
    action_reward_storage = RolloutStorage(train_config.num_envs)
    
    # tell the model to keep track of rollout memory
    model.eval()
    
    # reset and get first observation
    observation = env.reset()
    terminal = numpy.ones(train_config.num_envs, dtype=numpy.bool)
    reward = numpy.zeros(train_config.num_envs)
    
    with torch.no_grad():
        for step in tqdm.tqdm(
            range(train_config.batch_rollout_steps_per_epoch)
        ):
            # start new sequences if necessary
            action_reward_storage.start_new_seqs(terminal)
            observation_storage.start_new_seqs(terminal)
            
            # add latest observation to storage
            observation_storage.append_batch(observation=observation)
            
            # use model to compute actions
            i = torch.LongTensor(observation[:,0]).view(1,-1, 1).cuda()
            x = torch.LongTensor(observation[:,1]).view(1,-1).cuda()
            n, b = x.shape
            pad = (torch.ones(b, dtype=torch.long) * n).cuda()
            term = torch.BoolTensor(terminal).cuda()
            a_logits = model(x, i, pad, terminal=term)
            action_distribution = torch.distributions.Categorical(
                logits=a_logits[-1])
            a = action_distribution.sample().cpu().numpy()
            
            # send actions to the environment
            observation, reward, terminal, info = env.step(a)
            
            # make labels
            if train_config.env == 'next':
                labels = (x[0] + 1).cpu().numpy()
            elif train_config.env == 'count':
                labels, _ = observation_storage.get_current_seqs()
                labels = labels['observation'][:,:,1]
                labels = numpy.sum(labels, axis=0)
            elif train_config.env in ('max', 'variable_max'):
                labels, _ = observation_storage.get_current_seqs()
                labels = labels['observation'][:,:,1]
                labels = numpy.max(labels, axis=0)
            
            # store actions, rewards and labels
            action_reward_storage.append_batch(
                action=a,
                reward=reward,
                labels=labels,
            )
            
            # log
            log.add_scalar(
                'rollout/reward', numpy.sum(reward)/reward.shape[0], clock[0])
            clock[0] += 1
    
    return observation_storage | action_reward_storage

def train_epoch(
    train_config, epoch, model, optimizer, rollout_data, log, clock, debug=False
):
    print('-'*80)
    print('Training On Episodes')
    
    model.train()
    
    for p in range(1, train_config.training_passes_per_epoch+1):
        
        print('Pass: %i'%p)
        batch_iterator = rollout_data.batch_seq_iterator(
            train_config.batch_size, shuffle=True)
        for batch, pad in tqdm.tqdm(batch_iterator):
            
            # compute logits
            obs = batch['observation']
            i = torch.LongTensor(obs[:,:,0]).unsqueeze(-1).cuda()
            x = torch.LongTensor(obs[:,:,1]).cuda()
            n, b = x.shape
            pad = torch.LongTensor(pad).cuda()
            a_logits = model(x, i, pad).view(n*b,-1)
            
            # this can be used to verify parity between a whole-sequence forward
            # pass and a step-by-step forward pass
            if debug:
                b_logits = []
                with torch.no_grad():
                    model.eval()
                    steps = x.shape[0]
                    for step in range(steps):
                        xx = x[step].unsqueeze(0)
                        ii = i[step].unsqueeze(0)
                        if step == 0:
                            tt = torch.ones(b, dtype=torch.bool).cuda()
                        else:
                            tt = torch.zeros(b, dtype=torch.bool).cuda()
                        b_pad = torch.ones(b, dtype=torch.long).cuda()
                        b_logits.append(model(xx, ii, b_pad, terminal=tt))
                
                b_logits = torch.cat(b_logits, dim=0)
                import pdb
                pdb.set_trace()
            
            # compute loss
            a_labels = torch.LongTensor(batch['labels']).cuda().view(n*b)
            loss = torch.nn.functional.cross_entropy(
                a_logits, a_labels, reduction='none')
            loss = linearize_padded_seq(loss.view(n,b), pad)
            loss = torch.sum(loss) / torch.sum(pad)
            
            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # compute training accuracy
            a_prediction = torch.argmax(a_logits, dim=-1)
            train_correct = (a_prediction == a_labels).view(n,b)
            train_correct = linearize_padded_seq(train_correct, pad)
            train_correct = (
                torch.sum(train_correct).float() / torch.sum(pad))
            
            # log
            log.add_scalar('train/correct', train_correct, clock[0])
            clock[0] += 1
            

def save_checkpoint(train_config, epoch, model, optimizer, log, clock):
    frequency = train_config.checkpoint_frequency
    if frequency is not None and epoch % frequency == 0:
        checkpoint_directory = os.path.join(
            './checkpoint', os.path.split(log.log_dir)[-1])
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        
        print('-'*80)
        model_path = os.path.join(
            checkpoint_directory, 'model_%04i.pt'%epoch)
        print('Saving model to: %s'%model_path)
        torch.save(model.state_dict(), model_path)
        
        optimizer_path = os.path.join(
            checkpoint_directory, 'optimizer_%04i.pt'%epoch)
        print('Saving optimizer to: %s'%optimizer_path)
        torch.save(optimizer.state_dict(), optimizer_path)
