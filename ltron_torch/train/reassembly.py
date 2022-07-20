#!/usr/bin/env python
import random
import time
import os

import numpy

import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

from PIL import Image

from ltron.config import Config
from ltron.dataset.paths import get_dataset_info
from ltron.gym.envs.reassembly_env import reassembly_template_action
from ltron.gym.rollout_storage import RolloutStorage
from ltron.compression import batch_deduplicate_tiled_seqs
from ltron.hierarchy import (
    stack_numpy_hierarchies,
    len_hierarchy,
    index_hierarchy,
)
from ltron.visualization.drawing import write_text

from ltron_torch.models.padding import cat_padded_seqs, make_padding_mask
from ltron_torch.gym_tensor import gym_space_to_tensors, default_tile_transform
from ltron_torch.train.reassembly_labels import make_reassembly_labels
from ltron_torch.train.optimizer import adamw_optimizer
from ltron_torch.dataset.reassembly import (
    build_train_env,
    build_test_env,
    build_seq_train_loader,
)
from ltron_torch.models.reassembly import (
    build_reassembly_model,
    build_resnet_disassembly_model,
    #build_optimizer,
    observations_to_tensors,
    observations_to_resnet_tensors,
    #unpack_logits,
    logits_to_actions,
)


# config definitions ===========================================================

class StudentForcingReassemblyConfig(Config):
    epochs=10
    training_passes_per_epoch=8
    batch_size=16
    num_envs=16
    train_rollout_steps_per_epoch=4096
    test_rollout_steps_per_epoch=256
    max_episode_length=32
    
    disassembly_score=1.
    
    workspace_image_width=256
    workspace_image_height=256
    workspace_map_width=64
    workspace_map_height=64
    handspace_image_width=96
    handspace_image_height=96
    handspace_map_width=24
    handspace_map_height=24
    tile_width=16
    tile_height=16
    
    dataset='random_six'
    train_split='simple_single'
    train_subset=None
    test_split='simple_single'
    test_subset=None
    skip_reassembly=False
    
    test_frequency=None
    checkpoint_frequency=10
    visualization_frequency=1
    visualization_seqs=10
    
    def set_dependents(self):
        dataset_info = get_dataset_info(self.dataset)
        self.num_classes = max(dataset_info['shape_ids'].values()) + 1
        self.num_colors = max(dataset_info['color_ids'].values()) + 1
        
        self.train_batch_rollout_steps_per_epoch = (
            self.train_rollout_steps_per_epoch // self.num_envs
        )
        
        self.test_batch_rollout_steps_per_epoch = (
            seelf.test_rollout_steps_per_epoch // self.num_envs
        )


class TeacherForcingReassemblyConfig(Config):
    epochs=10
    batch_size=4
    num_envs=16
    loader_workers=8
    test_rollout_steps_per_epoch=256
    max_episode_length=64
    
    skip_reassembly=False
    
    learning_rate=3e-4
    weight_decay=0.1
    betas=(0.9, 0.95)
    
    disassembly_score=1.
    
    workspace_image_width=256
    workspace_image_height=256
    workspace_map_width=64
    workspace_map_height=64
    handspace_image_width=96
    handspace_image_height=96
    handspace_map_width=24
    handspace_map_height=24
    tile_width=16
    tile_height=16
    
    randomize_viewpoint=True
    randomize_colors=True
    
    dataset='random_six'
    train_split='simple_single_seq'
    train_subset=None
    test_split='simple_single'
    test_subset=None
    
    test_frequency=1
    checkpoint_frequency=10
    visualization_frequency=1
    visualization_seqs=10
    
    def set_dependents(self):
        dataset_info = get_dataset_info(self.dataset)
        self.num_classes = max(dataset_info['shape_ids'].values()) + 1
        self.num_colors = max(dataset_info['color_ids'].values()) + 1
        
        self.test_batch_rollout_steps_per_epoch = (
            self.test_rollout_steps_per_epoch // self.num_envs
        )


# train functions ==============================================================

def train_student_forcing_reassembly(train_config):
    
    print('='*80)
    print('Setup')
    print('-'*80)
    print('Log')
    log = SummaryWriter()
    clock = [0]
    
    model = build_reassembly_model(train_config)
    #optimizer = build_optimizer(train_config, model)
    optimizer = adamw_optimizer(model, train_config)
    train_env = build_train_env(train_config)
    test_env = None
    
    for epoch in range(1, train_config.epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        
        episodes = rollout_epoch(
            train_config, train_env, model, 'train', log, clock)
        visualize_episodes(train_config, epoch, episodes, log, clock)
        for p in range(1, num_passes+1):
            print('-'*80)
            print('Training pass: %i'%p)
            batch_iterator = episodes.batch_seq_iterator(
                train_config.batch_size, shuffle=True)
            train_pass(
                train_config, model, optimizer, batch_iterator, log, clock)
        save_checkpoint(train_config, epoch, model, optimizer, log, clock)
        #test_epoch(train_config, epoch, test_env, model, log, clock)
        
        epoch_end = time.time()
        print('-'*80)
        print('Epoch %i Elapsed: %.02f seconds'%(epoch, epoch_end-epoch_start))


def train_teacher_forcing_reassembly(train_config):
    
    print('='*80)
    print('Setup')
    print('-'*80)
    print('Log')
    log = SummaryWriter()
    clock = [0]
    train_start = time.time()
    
    model = build_reassembly_model(train_config)
    #optimizer = build_optimizer(train_config, model)
    optimizer = adamw_optimizer(model, train_config)
    train_loader = build_seq_train_loader(train_config)
    test_env = build_test_env(train_config)
    
    for epoch in range(1, train_config.epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        
        train_pass(train_config, model, optimizer, train_loader, log, clock)
        save_checkpoint(train_config, epoch, model, optimizer, log, clock)
        episodes = test_epoch(train_config, epoch, test_env, model, log, clock)
        visualize_episodes(train_config, epoch, episodes, log, clock)
        
        epoch_end = time.time()
        print('-'*80)
        print('Epoch %i Elapsed: %.02f seconds'%(epoch, epoch_end-epoch_start))
    
    train_end = time.time()
    print('='*80)
    print('Train elapsed: %.02f seconds'%(train_end-train_start))


def train_transformer_teacher_forcing_disassembly(train_config):
    
    print('='*80)
    print('Setup')
    print('-'*80)
    print('Log')
    log = SummaryWriter()
    clock = [0]
    train_start = time.time()
    
    train_config.skip_reassembly=True
    
    model = build_reassembly_model(train_config)
    #optimizer = build_optimizer(train_config, model)
    optimizer = adamw_optimizer(model, train_config)
    train_loader = build_seq_train_loader(train_config)
    test_env = build_test_env(train_config)
    
    for epoch in range(1, train_config.epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        
        train_pass(
            train_config, model, optimizer, train_loader, log, clock)
        save_checkpoint(train_config, epoch, model, optimizer, log, clock)
        episodes = test_transformer_disassembly_epoch(
            train_config, epoch, test_env, model, log, clock)
        visualize_episodes(train_config, epoch, episodes, log, clock)
        
        train_end = time.time()
        print('='*80)
        print('Train elapsed: %0.02f sections'%(train_end-train_start))


def train_resnet_teacher_forcing_disassembly(train_config):
    
    print('='*80)
    print('Setup')
    print('-'*80)
    print('Log')
    log = SummaryWriter()
    clock = [0]
    train_start = time.time()
    
    train_config.skip_reassembly=True
    
    model = build_resnet_disassembly_model(train_config)
    #optimizer = build_optimizer(train_config, model)
    optimizer = adamw_optimizer(model, train_config)
    train_loader = build_seq_train_loader(train_config)
    test_env = build_test_env(train_config)
    
    for epoch in range(1, train_config.epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        
        train_resnet_pass(
            train_config, model, optimizer, train_loader, log, clock)
        save_checkpoint(train_config, epoch, model, optimizer, log, clock)
        episodes = test_resnet_epoch(
            train_config, epoch, test_env, model, log, clock)
        visualize_episodes(train_config, epoch, episodes, log, clock)
        
        train_end = time.time()
        print('='*80)
        print('Train elapsed: %0.02f seconds'%(train_end-train_start))


# train subfunctions ===========================================================

def rollout_epoch(train_config, env, model, train_mode, log, clock):
    print('-'*80)
    print('Rolling out episodes')
    
    # initialize storage for observations, actions and rewards
    observation_storage = RolloutStorage(train_config.num_envs)
    action_reward_storage = RolloutStorage(train_config.num_envs)
    label_lists = [[] for _ in range(train_config.num_envs)]
    terminal_lists = []
    
    # tell the model to keep track of rollout memory
    model.eval()
    
    # use the train mode to determine the number of steps and rollout mode
    if train_mode == 'train':
        steps = train_config.train_batch_rollout_steps_per_epoch
        rollout_mode = 'sample'
    elif train_mode == 'test':
        steps = train_config.test_batch_rollout_steps_per_epoch
        rollout_mode = 'max'
    
    # reset and get first observation
    b = train_config.num_envs
    wh = train_config.workspace_image_height
    ww = train_config.workspace_image_width
    hh = train_config.handspace_image_height
    hw = train_config.handspace_image_width
    th = train_config.tile_height
    tw = train_config.tile_width
    prev_workspace_frame = numpy.ones((b, wh, ww, 3), dtype=numpy.uint8) * 102
    prev_handspace_frame = numpy.ones((b, hh, hw, 3), dtype=numpy.uint8) * 102
    
    # reset
    observation = env.reset()
    terminal = numpy.ones(train_config.num_envs, dtype=numpy.bool)
    reward = numpy.zeros(train_config.num_envs)
    
    with torch.no_grad():
        for step in tqdm.tqdm(range(steps)):
            # prep -------------------------------------------------------------
            # start new sequences if necessary
            action_reward_storage.start_new_seqs(terminal)
            observation_storage.start_new_seqs(terminal)
            terminal_lists.append(terminal)
            
            # get sequence lengths before adding the new observation
            seq_lengths = numpy.array(
                observation_storage.get_current_seq_lens())
                
            # add latest observation to storage
            observation_storage.append_batch(observation=observation)
            
            # make tiles -------------------------------------------------------
            # generate tile sequences from the workspace frame
            workspace_frame = observation['workspace_color_render'].reshape(
                1, b, wh, ww, 3)
            pad = numpy.ones(b, dtype=numpy.long)
            (wx, wi, w_pad) = batch_deduplicate_tiled_seqs(
                workspace_frame, pad, tw, th,
                background=prev_workspace_frame,
                s_start=seq_lengths,
            )
            prev_workspace_frame = workspace_frame
            wi = numpy.insert(wi, (0,3,3), -1, axis=-1)
            wi[:,:,0] = 0
            
            # generate tile sequences from the handspace frame
            handspace_frame = observation['handspace_color_render'].reshape(
                1, b, hh, hw, 3)
            (hx, hi, h_pad) = batch_deduplicate_tiled_seqs(
                handspace_frame, pad, tw, th,
                background=prev_handspace_frame,
                s_start=seq_lengths,
            )
            prev_handspace_frame = handspace_frame
            hi = numpy.insert(hi, (0,1,1), -1, axis=-1)
            hi[:,:,0] = 0
            
            # move tiles to torch and cuda
            wx = torch.FloatTensor(wx)
            hx = torch.FloatTensor(hx)
            w_pad = torch.LongTensor(w_pad)
            h_pad = torch.LongTensor(h_pad)
            tile_x, tile_pad = cat_padded_seqs(wx, hx, w_pad, h_pad)
            tile_x = default_tile_transform(tile_x).cuda()
            tile_pad = tile_pad.cuda()
            s, b = tile_x.shape[:2]
            tile_i, _ = cat_padded_seqs(
                torch.LongTensor(wi), torch.LongTensor(hi), w_pad, h_pad)
            tile_i = tile_i.cuda()
            
            # make tokens ------------------------------------------------------
            token_x = torch.LongTensor(
                observation['reassembly']['reassembling']).unsqueeze(0).cuda()
            token_i = torch.ones((1,b,6), dtype=torch.long) * -1
            token_i[:,:,0] = 0
            token_i[:,:,1] = torch.LongTensor(seq_lengths)
            token_i = token_i.cuda()
            token_pad = torch.ones(b, dtype=torch.long).cuda()
            
            # make decoder indices ---------------------------------------------
            decoder_i = torch.LongTensor(seq_lengths).unsqueeze(0).cuda()
            decoder_pad = torch.ones(b, dtype=torch.long).cuda()
            
            # make the terminal tensor -----------------------------------------
            terminal_tensor = torch.BoolTensor(terminal).cuda()
            
            # compute actions --------------------------------------------------
            action_logits, d_pad = model(
                tile_x=tile_x,
                tile_i=tile_i,
                tile_pad=tile_pad,
                token_x=token_x,
                token_i=token_i,
                token_pad=token_pad,
                decoder_i=decoder_i,
                decoder_pad=decoder_pad,
                terminal=terminal_tensor,
            )
            actions = logits_to_actions(
                action_logits,
                train_config.num_classes,
                train_config.num_colors,
                mode=rollout_mode,
            )
            
            '''
            mode_logits = action_logits['mode'].view(-1,5)
            mode_distribution = Categorical(logits=mode_logits)
            mode = mode_distribution.sample().cpu().numpy()
            
            polarity_logits = action_logits['disassemble_polarity'].view(-1,2)
            polarity_distribution = Categorical(logits=polarity_logits)
            polarity = polarity_distribution.sample().cpu().numpy()
            
            direction_logits = action_logits['disassemble_direction'].view(-1,2)
            direction_distribution = Categorical(logits=direction_logits)
            direction = direction_distribution.sample().cpu().numpy()
            
            pick_y_logits = action_logits['disassemble_pick_y'].view(-1,64)
            pick_y_distribution = Categorical(logits=pick_y_logits)
            pick_y = pick_y_distribution.sample().cpu().numpy()
            
            pick_x_logits = action_logits['disassemble_pick_x'].view(-1,64)
            pick_x_distribution = Categorical(logits=pick_x_logits)
            pick_x = pick_x_distribution.sample().cpu().numpy()
            
            pick = numpy.stack((pick_y, pick_x), axis=-1)
            
            # assemble actions
            actions = []
            for i in range(b):
                action = handspace_reassembly_template_action()
                action['reassembly'] = {
                    'start':mode[i] == 3,
                    'end':mode[i] == 4, 
                }
                action['disassembly'] = {
                    'activate':mode[i] == 0,
                    'polarity':polarity[i],
                    'direction':direction[i],
                    'pick':pick[i],
                }
                actions.append(action)
            '''
            
            # step -------------------------------------------------------------
            observation, reward, terminal, info = env.step(actions)
            
            '''
            # make labels ------------------------------------------------------
            for i, t in enumerate(terminal):
                if t or step == steps-1:
                    seq_id = observation_storage.batch_seq_ids[i]
                    labels = make_reassembly_labels(
                        observation_storage.get_seq(seq_id))
                    label_lists[i].extend(labels)
            '''
            
            # storage ----------------------------------------------------------
            action_reward_storage.append_batch(
                action=stack_numpy_hierarchies(*actions),
                reward=reward,
            )
    
    '''
    print('- '*40)
    print('Formatting Labels')
    # This is a goofy, hopefully temporary workaround for not having the
    # ability to insert entire sequences into a RolloutStorage instance.
    # The issue is that we generate labels for an entire sequence at a time
    # after the sequence has terminated, so we don't have the ability to use
    # append_batch at every step.  TODO: clean up.
    label_storage = RolloutStorage(train_config.num_envs)
    for i, terminal in enumerate(terminal_lists):
        frame_labels = [label_lists[j][i] for j in range(train_config.num_envs)]
        frame_labels = stack_numpy_hierarchies(*frame_labels)
        
        label_storage.start_new_seqs(terminal)
        label_storage.append_batch(label=frame_labels)
    '''
    
    return observation_storage | action_reward_storage # | label_storage


def rollout_resnet_epoch(train_config, env, model, train_mode, log, clock):
    print('-'*80)
    print('Rolling out episodes')
    
    # initialize storage for observations, actions and rewards
    observation_storage = RolloutStorage(train_config.num_envs)
    action_reward_storage = RolloutStorage(train_config.num_envs)
    label_lists = [[] for _ in range(train_config.num_envs)]
    terminal_lists = []
    
    # tell the model to keep track of rollout memory
    model.eval()
    
    # use the train mode to determine the number of steps and rollout mode
    if train_mode == 'train':
        steps = train_config.train_batch_rollout_steps_per_epoch
        rollout_mode = 'sample'
    elif train_mode == 'test':
        steps = train_config.test_batch_rollout_steps_per_epoch
        rollout_mode = 'max'
    
    # reset and get first observation
    b = train_config.num_envs
    wh = train_config.workspace_image_height
    ww = train_config.workspace_image_width
    hh = train_config.handspace_image_height
    hw = train_config.handspace_image_width
    th = train_config.tile_height
    tw = train_config.tile_width
    
    # reset
    observation = env.reset()
    terminal = numpy.ones(train_config.num_envs, dtype=numpy.bool)
    reward = numpy.zeros(train_config.num_envs)
    
    with torch.no_grad():
        for step in tqdm.tqdm(range(steps)):
            # prep -------------------------------------------------------------
            # start new sequences if necessary
            action_reward_storage.start_new_seqs(terminal)
            observation_storage.start_new_seqs(terminal)
            terminal_lists.append(terminal)
            
            # get sequence lengths before adding the new observation
            seq_lengths = numpy.array(
                observation_storage.get_current_seq_lens())
                
            # add latest observation to storage
            observation_storage.append_batch(observation=observation)
            
            # move tiles to torch and cuda
            b = env.num_envs
            pad = numpy.ones(b, dtype=numpy.long)
            observations = stack_numpy_hierarchies(observation)
            x = observations_to_resnet_tensors(train_config, observations, pad)
            x = x.cuda()
            
            # compute actions --------------------------------------------------
            xg, xd = model(x)
            s, b, c, h, w = xd.shape
            
            # build actions
            global_distribution = Categorical(logits=xg.view(-1, 9))
            mode_action = global_distribution.sample().cpu().numpy()
            actions = []
            for i in range(b):
                action = reassembly_template_action()
                if mode_action[i] == 0:
                    location_logits = xd[:,i,0,:,:].view(h*w)
                    location_distribution = Categorical(logits=location_logits)
                    location = location_distribution.sample().cpu().numpy()
                    y = location // w
                    x = location % w
                    
                    polarity_logits = xd[:,i,1:,y,x].view(2)
                    polarity_distribution = Categorical(logits=polarity_logits)
                    p = polarity_distribution.sample().cpu().numpy()
                    #import pdb
                    #pdb.set_trace()
                    action['workspace_cursor']['activate'] = 1
                    action['workspace_cursor']['position'] = numpy.array([y,x])
                    action['workspace_cursor']['polarity'] = p
                    action['disassembly'] = 1
                    
                elif mode_action[i] == 8:
                    action['reassembly'] = 1
                
                else:
                    action['workspace_viewpoint'] = mode_action[i]
                
                actions.append(action)
            
            # step -------------------------------------------------------------
            observation, reward, terminal, info = env.step(actions)
            
            # storage ----------------------------------------------------------
            action_reward_storage.append_batch(
                action=stack_numpy_hierarchies(*actions),
                reward=reward,
            )
    
    return observation_storage | action_reward_storage # | label_storage


def train_resnet_pass(train_config, model, optimizer, loader, log, clock):
    
    model.train()
    
    for batch, pad in tqdm.tqdm(loader):
        
        # convert observations to model tensors --------------------------------
        observations = batch['observations']
        x = observations_to_resnet_tensors(train_config, observations, pad)
        x = x.cuda()
        
        # forward --------------------------------------------------------------
        xg, xd = model(x)
        s, b, c, h, w = xd.shape
        
        # loss -----------------------------------------------------------------
        loss_mask = make_padding_mask(
            torch.LongTensor(pad), (s,b), mask_value=True).cuda()
        
        viewpoint_label = batch['actions']['workspace_viewpoint']
        end_label = (batch['actions']['reassembly'] == 1) * 8
        camera_label = torch.LongTensor(viewpoint_label + end_label).cuda()
        camera_loss = torch.nn.functional.cross_entropy(
            xg.view(-1,9), camera_label.view(-1), reduction='none').view(s,b)
        camera_loss = camera_loss * loss_mask
        camera_loss = torch.sum(camera_loss) / (s*b)
        log.add_scalar('train/camera_loss', camera_loss, clock[0])
        
        position_logits = xd[:,:,0].view(s*b, h*w)
        raw_position = batch['actions']['workspace_cursor']['position']
        position_target = torch.LongTensor(raw_position).cuda()
        position_target = position_target[:,:,0] * w + position_target[:,:,1]
        position_target = position_target.view(-1)
        position_loss = torch.nn.functional.cross_entropy(
            position_logits, position_target, reduction='none').view(s,b)
        position_mask = loss_mask * (camera_label == 0)
        position_loss = position_loss * position_mask
        position_loss = torch.sum(position_loss) / (s*b)
        log.add_scalar('train/position_loss', position_loss, clock[0])
        
        polarity_logits = xd[:,:,1:].view(s*b, 2, h, w)
        y = raw_position[:,:,0].reshape(-1)
        x = raw_position[:,:,1].reshape(-1)
        polarity_logits = polarity_logits[range(s*b), :, y, x]
        polarity_target = torch.LongTensor(
            batch['actions']['workspace_cursor']['polarity']).view(-1).cuda()
        polarity_loss = torch.nn.functional.cross_entropy(
            polarity_logits, polarity_target, reduction='none').view(s,b)
        polarity_loss = polarity_loss * position_mask
        polarity_loss = torch.sum(polarity_loss) / (s*b)
        log.add_scalar('train/polarity_loss', polarity_loss, clock[0])
        
        loss = camera_loss + position_loss + polarity_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        clock[0] += 1

def train_pass(train_config, model, optimizer, loader, log, clock):
    
    model.train()
    
    for batch, pad in tqdm.tqdm(loader):
        
        b = pad.shape[0]
        
        # convert observations to model tensors --------------------------------
        observations = batch['observations']
        (tile_x, tile_i, tile_pad,
         token_x, token_i, token_pad,
         decoder_i, decoder_pad) = observations_to_tensors(
            train_config, observations, pad)
        
        # compute logits -------------------------------------------------------
        logits, d_pad = model(
            tile_x=tile_x,
            tile_i=tile_i,
            tile_pad=tile_pad,
            token_x=token_x,
            token_i=token_i,
            token_pad=token_pad,
            decoder_i=decoder_i,
            decoder_pad=decoder_pad,
        )
        
        #action_logits = unpack_logits(
        #    action_logits, train_config.num_classes, train_config.num_colors)
        
        # compute losses -------------------------------------------------------
        s, b = logits['disassembly'].shape[:2]
        mask = make_padding_mask(d_pad, (s,b), mask_value=True).cuda()
        labels = batch['actions']
        
        loss = 0
        loss = loss + viewpoint_loss('workspace', logits, labels, mask)
        loss = loss + viewpoint_loss('handspace', logits, labels, mask)
        loss = loss + cursor_loss('workspace', logits, labels, mask)
        loss = loss + cursor_loss('handspace', logits, labels, mask)
        loss = loss + disassembly_loss(logits, labels, mask)
        loss = loss + insert_brick_loss(logits, labels, mask)
        loss = loss + pick_and_place_loss(logits, labels, mask)
        loss = loss + rotate_loss(logits, labels, mask)
        loss = loss + reassembly_loss(logits, labels, mask)
        log.add_scalar('train/loss', loss, clock[0])
        clock[0] += 1
        
        # optimize ---------------------------------------------------------
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_resnet_epoch(train_config, epoch, test_env, model, log, clock):
    frequency = train_config.test_frequency
    if frequency is not None and epoch % frequency == 0:
        episodes = rollout_resnet_epoch(
            train_config, test_env, model, 'test', log, clock)
        
        avg_terminal_reward = 0.
        for seq_id in episodes.finished_seqs:
            seq = episodes.get_seq(seq_id)
            avg_terminal_reward += seq['reward'][-1]
        
        if episodes.num_finished_seqs():
            avg_terminal_reward /= episodes.num_finished_seqs()
        
        print('Average Terminal Reward: %f'%avg_terminal_reward)
        log.add_scalar('val/term_reward', avg_terminal_reward, clock[0])
        
        avg_reward = 0.
        entries = 0
        for seq_id in range(episodes.num_seqs()):
            seq = episodes.get_seq(seq_id)
            avg_reward += numpy.sum(seq['reward'])
            entries += seq['reward'].shape[0]
        
        avg_reward /= entries
        print('Average Reward: %f'%avg_reward)
        log.add_scalar('val/reward', avg_reward, clock[0])
        
        return episodes
    
    else:
        return None

def test_epoch(train_config, epoch, test_env, model, log, clock):
    frequency = train_config.test_frequency
    if frequency is not None and epoch % frequency == 0:
        episodes = rollout_epoch(
            train_config, test_env, model, 'test', log, clock)
        #for seq in range(episodes.num_seqs()):
        #    print(episodes.get_seq(seq)['reward'])
        
        avg_terminal_reward = 0.
        for seq_id in episodes.finished_seqs:
            seq = episodes.get_seq(seq_id)
            avg_terminal_reward += seq['reward'][-1]
        avg_terminal_reward /= episodes.num_finished_seqs()
        
        print('Reward: %f'%avg_terminal_reward)
        
        return episodes
    
    else:
        return None

def visualize_episodes(train_config, epoch, episodes, log, clock):
    frequency = train_config.visualization_frequency
    if epoch % frequency == 0:
        print('-'*80)
        print('Generating Visualizations')
        
        visualization_directory = os.path.join(
            './visualization',
            os.path.split(log.log_dir)[-1],
            'epoch_%04i'%epoch,
        )
        if not os.path.exists(visualization_directory):
            os.makedirs(visualization_directory)
        
        num_seqs = min(train_config.visualization_seqs, episodes.num_seqs())
        for seq_id in tqdm.tqdm(range(num_seqs)):
            seq_path = os.path.join(visualization_directory, 'seq_%06i'%seq_id)
            if not os.path.exists(seq_path):
                os.makedirs(seq_path)
            
            seq = episodes.get_seq(seq_id)
            seq_len = len_hierarchy(seq)
            workspace_frames = seq['observation']['workspace_color_render']
            handspace_frames = seq['observation']['handspace_color_render']
            for frame_id in range(seq_len):
                workspace_frame = workspace_frames[frame_id]
                handspace_frame = handspace_frames[frame_id]
                wh, ww = workspace_frame.shape[:2]
                hh, hw = handspace_frame.shape[:2]
                w = ww + hw
                joined_image = numpy.zeros((wh, w, 3), dtype=numpy.uint8)
                joined_image[:,:ww] = workspace_frame
                joined_image[wh-hh:,ww:] = handspace_frame
                
                '''
                def mode_string(action):
                    if action['disassembly']:
                        return 'Disassembly'
                    elif action['insert_brick']['class_id'] != 0:
                        return 'Insert Brick [%i] [%i]'%(
                            action['insert_brick']['class_id'],
                            action['insert_brick']['color_id'])
                    elif action['pick_and_place']['activate']:
                        polarity = action['pick_and_place']['polarity']
                        return 'PickAndPlace [%i]'%polarity
                    elif action['rotate']['activate']:
                        return 'Rotate'
                    elif action['reassembly']['start']:
                        return 'StartReassembly'
                    elif action['reassembly']['end']:
                        return 'EndEpisode'
                '''
                def action_string(action):
                    result = []
                    if action['disassembly']:
                        result.append('Disassembly')
                    if action['insert_brick']['class_id'] != 0:
                        result.append('Insert Brick [%i] [%i]'%(
                            action['insert_brick']['class_id'],
                            action['insert_brick']['color_id'],
                        ))
                    if action['pick_and_place'] == 1:
                        result.append('PickAndPlaceCursor')
                    if action['pick_and_place'] == 2:
                        result.append('PickAndPlaceOrigin')
                    if action['rotate']:
                        result.append('Rotate [%i]'%action['rotate'])
                    return '\n'.join(result)
                
                def draw_workspace_dot(position, color):
                    y, x = position
                    joined_image[y*4:(y+1)*4, x*4:(x+1)*4] = color
                
                def draw_handspace_dot(position, color):
                    y, x = position
                    yy = wh - hh
                    joined_image[yy+y*4:yy+(y+1)*4, ww+x*4:ww+(x+1)*4] = color
                
                def draw_pick_and_place(action):
                    if action['disassembly']:
                        pick = action['workspace_cursor']['position']
                        polarity = action['workspace_cursor']['polarity']
                        if polarity:
                            color = (0,0,255)
                        else:
                            color = (255,0,0)
                        draw_workspace_dot(pick, color)
                    
                    if action['pick_and_place']:
                        pick = action['handspace_cursor']['position']
                        draw_handspace_dot(pick, color)
                        place = action['workspace_cursor']['position']
                        draw_workspace_dot(place, color)
                
                lines = []
                frame_action = index_hierarchy(seq['action'], frame_id)
                #frame_label = index_hierarchy(seq['label'], frame_id)
                lines.append('Model:\n' + action_string(frame_action) + '\n')
                #lines.append('Label: %s'%mode_string(frame_label))
                lines.append('Reward: %f'%seq['reward'][frame_id])
                joined_image = write_text(joined_image, '\n'.join(lines))
                draw_pick_and_place(frame_action)
                #draw_pick_and_place(frame_label, (0,255,0))
                
                frame_path = os.path.join(
                    seq_path,
                    'frame_%04i_%06i_%04i.png'%(epoch, seq_id, frame_id),
                )
                Image.fromarray(joined_image).save(frame_path)


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


# loss functions ===============================================================
def masked_cross_entropy(logits, labels, mask):
    d = logits.shape[2]
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, d), labels.view(-1), reduction='none')
    loss = loss * mask.view(-1)
    loss = torch.sum(loss) / loss.numel()
    return loss

def viewpoint_loss(name, logits, labels, mask):
    component_name = '%s_viewpoint'%name
    logits = logits[component_name]
    labels = torch.LongTensor(labels[component_name]).cuda()
    return masked_cross_entropy(logits, labels, mask)
    
def cursor_loss(name, logits, labels, mask):
    loss = 0
    component_name = '%s_cursor'%name
    
    activate_logits = logits[component_name + '_activate']
    activate_labels = torch.LongTensor(
        labels[component_name]['activate']).cuda()
    loss = loss + masked_cross_entropy(activate_logits, activate_labels, mask)
    
    activate_mask = mask * activate_labels
    for i, n in (0, 'y'), (1, 'x'):
        position_logits = logits[component_name + '_' + n]
        position_labels = torch.LongTensor(
            labels[component_name]['position'][:,:,i]).cuda()
        loss = loss + masked_cross_entropy(
            position_logits, position_labels, activate_mask)
    
    polarity_logits = logits[component_name + '_p']
    polarity_labels = torch.LongTensor(
            labels[component_name]['polarity']).cuda()
    loss = loss + masked_cross_entropy(
        polarity_logits, polarity_labels, activate_mask)
    
    return loss

def disassembly_loss(logits, labels, mask):
    disassembly_logits = logits['disassembly']
    disassembly_labels = torch.LongTensor(labels['disassembly']).cuda()
    return masked_cross_entropy(disassembly_logits, disassembly_labels, mask)

def insert_brick_loss(logits, labels, mask):
    labels = labels['insert_brick']
    s, b = labels['class_id'].shape[:2]
    num_classes = logits['insert_brick_class'].shape[-1]
    num_colors = logits['insert_brick_color'].shape[-1]
    
    class_id_logits = logits['insert_brick_class']
    class_id_labels = torch.LongTensor(labels['class_id']).cuda()
    class_id_loss = masked_cross_entropy(class_id_logits, class_id_labels, mask)
    #class_id_loss = torch.nn.functional.cross_entropy(
    #    class_id_logits, class_id_labels.view(-1), reduction='none')
    
    color_id_logits = logits['insert_brick_color']
    color_id_labels = torch.LongTensor(labels['color_id']).cuda()
    color_id_loss = masked_cross_entropy(color_id_logits, color_id_labels, mask)
    #color_id_loss = torch.nn.functional.cross_entropy(
    #    color_id_logits, color_id_labels.view(-1), reduction='none')
    
    #insert_brick_loss = class_id_loss + color_id_loss
    #insert_brick_loss = insert_brick_loss.view(s, b)
    #insert_brick_loss = insert_brick_loss * make_padding_mask(
    #    decoder_pad, (s,b), mask_value=True).cuda()
    #insert_brick_loss = (
    #    torch.sum(insert_brick_loss) / insert_brick_loss.numel())
    
    #return insert_brick_loss
    return class_id_loss + color_id_loss

def pick_and_place_loss(logits, labels, mask):
    pick_and_place_logits = logits['pick_and_place']
    pick_and_place_labels = torch.LongTensor(labels['pick_and_place']).cuda()
    #print(pick_and_place_labels)
    #print(pick_and_place_logits)
    return masked_cross_entropy(
        pick_and_place_logits, pick_and_place_labels, mask)

def rotate_loss(logits, labels, mask):
    rotate_logits = logits['rotate']
    rotate_labels = torch.LongTensor(labels['rotate']).cuda()
    return masked_cross_entropy(rotate_logits, rotate_labels, mask)

def reassembly_loss(logits, labels, mask):
    reassembly_logits = logits['reassembly']
    reassembly_labels = torch.LongTensor(labels['reassembly']).cuda()
    return masked_cross_entropy(reassembly_logits, reassembly_labels, mask)

def old_mode_loss(action_logits, labels, decoder_pad):
    mode_logits = action_logits['mode']
    s,b = mode_logits.shape[:2]
    #import pdb
    #pdb.set_trace()
    mode_labels = torch.LongTensor(
        labels['disassembly']['activate'].astype(numpy.long) *
        numpy.ones((s, b), dtype=numpy.long) * 0 +
        # ----
        (labels['insert_brick']['class_id'] != 0).astype(numpy.long) *
        numpy.ones((s, b), dtype=numpy.long) * 1 +
        # ----
        labels['pick_and_place']['activate'].astype(numpy.long) *
        numpy.ones((s, b), dtype=numpy.long) * 2 +
        # ----
        (labels['rotate'] == 1).astype(numpy.long) *
        numpy.ones((s, b), dtype=numpy.long) * 3 +
        # ----
        (labels['rotate'] == 2).astype(numpy.long) *
        numpy.ones((s, b), dtype=numpy.long) * 4 +
        # ----
        (labels['rotate'] == 3).astype(numpy.long) *
        numpy.ones((s, b), dtype=numpy.long) * 5 +
        # ----
        labels['reassembly']['start'].astype(numpy.long) *
        numpy.ones((s, b), dtype=numpy.long) * 6 +
        # ----
        labels['reassembly']['end'].astype(numpy.long) *
        numpy.ones((s, b), dtype=numpy.long) * 7
        
    ).cuda()
    mode_labels = mode_labels.view(-1)
    mode_loss = torch.nn.functional.cross_entropy(
        mode_logits.view(-1,8), mode_labels.view(-1))
    
    return mode_loss


def old_disassembly_loss(action_logits, labels, decoder_pad):
    #labels = labels['disassembly']
    cursor_labels = labels['workspace_cursor']
    activate = labels['disassembly']['activate']
    s, b = activate.shape
    if numpy.any(activate):
        polarity_logits = action_logits['disassemble_polarity'].view(-1,2)
        #polarity_labels = torch.LongTensor(labels['polarity']).cuda()
        polarity_labels = torch.LongTensor(cursor_labels['polarity']).cuda()
        polarity_loss = torch.nn.functional.cross_entropy(
            polarity_logits, polarity_labels.view(-1),
            reduction='none')
        
        #direction_logits = action_logits['disassemble_direction'].view(-1,2)
        #direction_labels = torch.LongTensor(labels['direction']).cuda()
        #direction_loss = torch.nn.functional.cross_entropy(
        #    direction_logits, direction_labels.view(-1),
        #    reduction='none')
        
        pick_y_logits = action_logits['disassemble_pick_y'].view(-1,64)
        #pick_y_labels = torch.LongTensor(labels['pick'][:,:,0]).cuda()
        pick_y_labels = torch.LongTensor(
            cursor_labels['position'][:,:,0]).cuda()
        pick_y_loss = torch.nn.functional.cross_entropy(
            pick_y_logits, pick_y_labels.view(-1),
            reduction='none')
        
        pick_x_logits = action_logits['disassemble_pick_x'].view(-1,64)
        #pick_x_labels = torch.LongTensor(labels['pick'][:,:,1]).cuda()
        pick_x_labels = torch.LongTensor(
            cursor_labels['position'][:,:,1]).cuda()
        pick_x_loss = torch.nn.functional.cross_entropy(
            pick_x_logits, pick_x_labels.view(-1),
            reduction='none')
        
        disassembly_loss = (
            polarity_loss +
            #direction_loss +
            pick_y_loss +
            pick_x_loss
        )
        disassembly_loss = disassembly_loss.view(s, b)
        disassembly_loss = disassembly_loss * make_padding_mask(
            decoder_pad, (s,b), mask_value=True).cuda()
        disassembly_loss = disassembly_loss * torch.BoolTensor(activate).cuda()
        disassembly_loss = (
            torch.sum(disassembly_loss) / disassembly_loss.numel())
    else:
        disassembly_loss = 0.
    
    return disassembly_loss


def old_insert_brick_loss(action_logits, labels, decoder_pad):
    labels = labels['insert_brick']
    s, b = labels['class_id'].shape[:2]
    num_classes = action_logits['insert_brick_class_id'].shape[-1]
    num_colors = action_logits['insert_brick_color_id'].shape[-1]
    
    class_id_logits = action_logits['insert_brick_class_id'].view(
        -1, num_classes)
    class_id_labels = torch.LongTensor(labels['class_id']).cuda()
    class_id_loss = torch.nn.functional.cross_entropy(
        class_id_logits, class_id_labels.view(-1),
        reduction='none')
    
    color_id_logits = action_logits['insert_brick_color_id'].view(
        -1, num_colors)
    color_id_labels = torch.LongTensor(labels['color_id']).cuda()
    color_id_loss = torch.nn.functional.cross_entropy(
        color_id_logits, color_id_labels.view(-1),
        reduction='none')
    
    insert_brick_loss = class_id_loss + color_id_loss
    insert_brick_loss = insert_brick_loss.view(s, b)
    insert_brick_loss = insert_brick_loss * make_padding_mask(
        decoder_pad, (s,b), mask_value=True).cuda()
    insert_brick_loss = (
        torch.sum(insert_brick_loss) / insert_brick_loss.numel())
    
    return insert_brick_loss


def old_pick_and_place_loss(action_logits, labels, decoder_pad):
    #labels = labels['pick_and_place']
    p_labels = labels['pick_and_place']
    h_cursor_labels = labels['handspace_cursor']
    w_cursor_labels = labels['workspace_cursor']
    activate = labels['pick_and_place']['activate']
    s, b = activate.shape
    if numpy.any(activate):
        polarity_logits = action_logits['pick_and_place_polarity'].view(-1,2)
        #polarity_labels = torch.LongTensor(labels['polarity']).cuda()
        polarity_labels = torch.LongTensor(h_cursor_labels['polarity']).cuda()
        polarity_loss = torch.nn.functional.cross_entropy(
            polarity_logits, polarity_labels.view(-1),
            reduction='none')
        
        at_origin_logits = action_logits['pick_and_place_at_origin'].view(-1,2)
        at_origin_labels = torch.LongTensor(p_labels['place_at_origin']).cuda()
        at_origin_loss = torch.nn.functional.cross_entropy(
            at_origin_logits, at_origin_labels.view(-1),
            reduction='none')
        
        pick_y_logits = action_logits['pick_and_place_pick_y'].view(-1,24)
        #pick_y_labels = torch.LongTensor(labels['pick'][:,:,0]).cuda()
        pick_y_labels = torch.LongTensor(
            h_cursor_labels['position'][:,:,0]).cuda()
        pick_y_loss = torch.nn.functional.cross_entropy(
            pick_y_logits, pick_y_labels.view(-1),
            reduction='none')
        
        pick_x_logits = action_logits['pick_and_place_pick_x'].view(-1,24)
        #pick_x_labels = torch.LongTensor(labels['pick'][:,:,1]).cuda()
        pick_x_labels = torch.LongTensor(
            h_cursor_labels['position'][:,:,1]).cuda()
        pick_x_loss = torch.nn.functional.cross_entropy(
            pick_x_logits, pick_x_labels.view(-1),
            reduction='none')
        
        place_y_logits = action_logits['pick_and_place_place_y'].view(-1,64)
        #place_y_labels = torch.LongTensor(labels['place'][:,:,0]).cuda()
        place_y_labels = torch.LongTensor(
            w_cursor_labels['position'][:,:,0]).cuda()
        place_y_loss = torch.nn.functional.cross_entropy(
            place_y_logits, place_y_labels.view(-1),
            reduction='none')
        
        place_x_logits = action_logits['pick_and_place_place_x'].view(-1,64)
        #place_x_labels = torch.LongTensor(labels['place'][:,:,1]).cuda()
        place_x_labels = torch.LongTensor(
            w_cursor_labels['position'][:,:,1]).cuda()
        place_x_loss = torch.nn.functional.cross_entropy(
            place_x_logits, place_x_labels.view(-1),
            reduction='none')
        
        pick_and_place_loss = (
            polarity_loss +
            at_origin_loss +
            pick_y_loss +
            pick_x_loss +
            place_y_loss +
            place_x_loss
        )
        pick_and_place_loss = pick_and_place_loss.view(s, b)
        pick_and_place_loss = pick_and_place_loss * make_padding_mask(
            decoder_pad, (s,b), mask_value=True).cuda()
        pick_and_place_loss = (
            pick_and_place_loss * torch.BoolTensor(activate).cuda())
        pick_and_place_loss = (
            torch.sum(pick_and_place_loss) / pick_and_place_loss.numel())
    else:
        pick_and_place_loss = 0.
    
    return pick_and_place_loss


def old_rotate_loss(action_logits, labels, decoder_pad):
    return 0.
