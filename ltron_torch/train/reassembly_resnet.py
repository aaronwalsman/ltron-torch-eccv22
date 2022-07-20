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
from ltron_torch.models.reassembly_resnet import (
    build_model as build_resnet_model,
)
from ltron_torch.models.reassembly_lstm import build_model as build_lstm_model
from ltron_torch.envs.reassembly_lstm_interface import ReassemblyLSTMInterface
from ltron_torch.train.behavior_cloning import behavior_cloning

# config definitions ===========================================================

class BehaviorCloningReassemblyConfig(Config):
    epochs=10
    batch_size=4
    num_envs=16
    loader_workers=8
    test_rollout_steps_per_epoch=256
    max_episode_length=64
    
    skip_reassembly=False
    insert_only=False
    
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
    
    randomize_viewpoint=True,
    randomize_colors=True,
    
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
        self.num_classes = max(dataset_info['class_ids'].values()) + 1
        self.num_colors = max(dataset_info['color_ids'].values()) + 1
        
        self.test_batch_rollout_steps_per_epoch = (
            self.test_rollout_steps_per_epoch // self.num_envs
        )


# train functions ==============================================================

def train_disassembly_behavior_cloning(train_config):
    print('='*80)
    print('Disassembly Behavior Cloning Setup')
    # make everything
    train_config.skip_reassembly=True
    model = build_resnet_model(train_config)
    optimizer = adamw_optimizer(model, train_config)
    train_loader = build_seq_train_loader(train_config)
    test_env = build_test_env(train_config)
    interface = ReassemblyLSTMInterface(train_config)
    
    # run behavior cloning
    behavior_cloning(
        train_config, model, optimizer, train_loader, test_env, interface)

def train_disassembly_behavior_cloning_old(train_config):
    
    print('='*80)
    print('Setup')
    print('-'*80)
    print('Log')
    log = SummaryWriter()
    clock = [0]
    train_start = time.time()
    
    train_config.skip_reassembly=True
    
    model = build_resnet_model(train_config)
    #model = build_lstm_model(train_config)
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
        episodes = test_epoch(
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
            
            # move observations to torch and cuda
            b = env.num_envs
            pad = numpy.ones(b, dtype=numpy.long)
            observations = stack_numpy_hierarchies(observation)
            x = observations_to_tensors(train_config, observations, pad)
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
                    
                    action['workspace_cursor']['activate'] = 1
                    action['workspace_cursor']['position'] = numpy.array([y,x])
                    action['workspace_cursor']['polarity'] = p
                    action['disassembly'] = 1
                    
                elif mode_action[i] == 8:
                    action['reassembly'] = 1
                
                else:
                    action['workspace_viewpoint'] = mode_action[i]
                
                actions.append(action)
            
            location_p = torch.softmax(
                xd[:,:,0].view(b, h*w), dim=-1).view(b, h, w)
            polarity_p = torch.softmax(
                xd[:,:,1:].view(b, 2, h*w), dim=-2).view(b, 2, h, w)
            
            # step -------------------------------------------------------------
            observation, reward, terminal, info = env.step(actions)
            
            # storage ----------------------------------------------------------
            action_reward_storage.append_batch(
                action=stack_numpy_hierarchies(*actions),
                reward=reward,
                location_p = location_p.detach().cpu().numpy(),
                polarity_p = polarity_p.detach().cpu().numpy(),
            )
    
    return observation_storage | action_reward_storage # | label_storage


def train_pass(train_config, model, optimizer, loader, log, clock):
    
    model.train()
    
    for batch, pad in tqdm.tqdm(loader):
        
        # convert observations to model tensors --------------------------------
        observations = batch['observations']
        xw, xh = observations_to_tensors(train_config, observations, pad)
        #xw, xh = xw.cuda(), xh.cuda()
        xw = xw.cuda()
        
        # forward --------------------------------------------------------------
        #xg, xd = model(xw, xh)
        xg, xd = model(xw)
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


def test_epoch(train_config, epoch, test_env, model, log, clock):
    frequency = train_config.test_frequency
    if frequency is not None and epoch % frequency == 0:
        episodes = rollout_epoch(
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
                
                def draw_workspace_dot(position, color, alpha=1.0):
                    y, x = position
                    joined_image[y*4:(y+1)*4, x*4:(x+1)*4] = (
                        color * alpha +
                        joined_image[y*4:(y+1)*4, x*4:(x+1)*4] * (1. - alpha)
                    )
                
                def draw_handspace_dot(position, color):
                    y, x = position
                    yy = wh - hh
                    joined_image[yy+y*4:yy+(y+1)*4, ww+x*4:ww+(x+1)*4] = (
                        color * alpha +
                        joined_image[yy+y*4:yy+(y+1)*4, ww+x*4:ww+(x+1)*4] *
                            (1. - alpha)
                    )
                
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
                lines.append('Model:\n' + action_string(frame_action) + '\n')
                lines.append('Reward: %f'%seq['reward'][frame_id])
                joined_image = write_text(joined_image, '\n'.join(lines))
                #draw_pick_and_place(frame_action)
                
                for y in range(64):
                    for x in range(64):
                        pyx = seq['location_p'][frame_id, y, x]
                        pp = seq['polarity_p'][frame_id, :, y, x]
                        color = (
                            pp[0] * numpy.array([255,0,0]) +
                            pp[1] * numpy.array([0,0,255])
                        )
                        draw_workspace_dot((y, x), color, pyx*5.)
                
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
