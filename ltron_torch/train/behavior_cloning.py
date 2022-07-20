#!/usr/bin/env python
import time
import os

import numpy

import tqdm

import torch
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
#from torch.utils.tensorboard import SummaryWriter

from splendor.image import save_image

from ltron.config import Config
from ltron.dataset.paths import get_dataset_info
from ltron.gym.rollout_storage import RolloutStorage
from ltron.hierarchy import (
    stack_numpy_hierarchies,
    len_hierarchy,
    index_hierarchy,
)
from ltron.visualization.drawing import write_text

from ltron_torch.models.padding import make_padding_mask
from ltron_torch.train.reassembly_labels import make_reassembly_labels
from ltron_torch.train.optimizer import clip_grad

# config definitions ===========================================================

class BehaviorCloningConfig(Config):
    epochs = 10
    batch_size = 4
    num_test_envs = 4
    
    train_frequency = 1
    test_frequency = 1
    checkpoint_frequency = 10
    visualization_frequency = 1
    
    #test_rollout_steps_per_epoch = 2048
    test_episodes_per_epoch = 1024
    
    checkpoint_directory = './checkpoint'
    
    #def set_dependents(self):
    #    self.test_batch_rollout_steps_per_epoch = (
    #        self.test_rollout_steps_per_epoch // self.num_test_envs
    #    )


# train functions ==============================================================

def behavior_cloning(
    config,
    model,
    optimizer,
    scheduler,
    train_loader,
    test_env,
    interface,
    train_log,
    test_log,
    start_epoch = 1,
):
    
    print('='*80)
    print('Begin Behavior Cloning')
    
    train_start = time.time()
    
    for epoch in range(start_epoch, config.epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        
        train_freq = config.train_frequency
        if train_freq and epoch % train_freq == 0:
            train_epoch(
                config,
                epoch,
                model,
                optimizer,
                scheduler,
                train_loader,
                interface,
                train_log,
            )
            chart = train_log.plot_grid(
                topline=True, legend=True, minmax_y=True, height=40, width=72)
            print(chart)
        
        checkpoint_freq = config.checkpoint_frequency
        if checkpoint_freq and epoch % checkpoint_freq == 0:
            save_checkpoint(
                config, epoch, model, optimizer, scheduler, train_log, test_log)
        
        test_freq = config.test_frequency
        test = test_freq and epoch % test_freq == 0
        vis_freq = config.visualization_frequency
        visualize = vis_freq and epoch % vis_freq == 0
        
        if test or visualize:
            episodes = rollout_epoch(
                epoch,
                config,
                test_env,
                model,
                interface,
                'test',
                True,
                visualize,
            )
        
        if test:
            test_episodes(config, epoch, episodes, interface, test_log)
            test_log.step()
            chart = test_log.plot_sequential(
                legend=True, minmax_y=True, height=60, width=160)
            print(chart)
        
        if visualize:
            visualize_episodes(config, epoch, episodes, interface)
        
        train_end = time.time()
        print('='*80)
        print('Train elapsed: %0.02f seconds'%(train_end-train_start))


# train subfunctions ===========================================================

def train_epoch(
    config,
    epoch,
    model,
    optimizer,
    scheduler,
    loader,
    interface,
    train_log,
):
    print('-'*80)
    print('Training')
    model.train()
    
    for batch, pad in tqdm.tqdm(loader):
        
        observations = batch['observations']
        actions = batch['actions']
        
        # convert observations to model tensors
        x = interface.observation_to_tensors(observations, actions, pad)
        y = interface.action_to_tensors(actions, pad)
        
        if hasattr(interface, 'augment'):
            x, y = interface.augment(x, y)
        
        # forward
        x = model(**x)
        
        # loss
        loss = interface.loss(x, y, pad, train_log)
        
        # train
        optimizer.zero_grad()
        loss.backward()
        clip_grad(config, model)
        scheduler.step()
        optimizer.step()
        
        train_log.step()

def rollout_epoch(
    epoch,
    config,
    env,
    model,
    interface,
    train_mode,
    store_observations,
    store_activations,
):
    print('-'*80)
    print('Rolling out episodes')
    
    # initialize storage for observations, actions, rewards and activations
    if train_mode == 'test':
        num_envs = config.num_test_envs
    elif train_mode == 'train':
        num_envs = config.num_train_envs
    if store_observations:
        observation_storage = RolloutStorage(num_envs)
    action_reward_storage = RolloutStorage(num_envs)
    
    if store_activations:
        activation_storage = RolloutStorage(num_envs)
    
    # put the model in eval mode
    model.eval()
    device = next(model.parameters()).device
    
    # use the train mode to determine the number of steps and rollout mode
    if train_mode == 'train':
        raise NotImplementedError
        steps = config.train_batch_rollout_steps_per_epoch
        rollout_mode = 'sample'
    elif train_mode == 'test':
        episodes = config.test_episodes_per_epoch
        rollout_mode = 'max'
    b = num_envs
    
    # reset
    observation = env.reset()
    terminal = numpy.ones(num_envs, dtype=numpy.bool)
    reward = numpy.zeros(num_envs)
    
    with torch.no_grad():
        if hasattr(model, 'initialize_memory'):
            memory = model.initialize_memory(b)
        else:
            memory = None
        
        #for step in tqdm.tqdm(range(steps)):
        progress = tqdm.tqdm(total=episodes)
        with progress:
            while action_reward_storage.num_finished_seqs() < episodes:
                # prep ---------------------------------------------------------
                # start new sequences if necessary
                action_reward_storage.start_new_seqs(terminal)
                if store_observations:
                    observation_storage.start_new_seqs(terminal)
                if store_activations:
                    activation_storage.start_new_seqs(terminal)
                
                # add latest observation to storage
                if store_observations:
                    observation_storage.append_batch(observation=observation)
                
                # move observations to torch and cuda
                pad = numpy.ones(b, dtype=numpy.long)
                observation = stack_numpy_hierarchies(observation)
                x = interface.observation_to_tensors(observation, None, pad)
                
                # compute actions ----------------------------------------------
                if hasattr(model, 'initialize_memory'):
                    if hasattr(interface, 'forward_rollout'):
                        x = interface.forward_rollout(
                            terminal, **x, memory=memory)
                    else:
                        x = model(**x, memory=memory)
                    memory = x['memory']
                else:
                    if hasattr(interface, 'forward_rollout'):
                        x = interface.forward_rollout(terminal, **x)
                    else:
                        x = model(**x)
                actions = interface.tensor_to_actions(x, env, mode=rollout_mode)
                
                # step ---------------------------------------------------------
                observation, reward, terminal, info = env.step(actions)
                
                # reset memory -------------------------------------------------
                if hasattr(model, 'reset_memory'):
                    model.reset_memory(memory, terminal)
                
                # storage ------------------------------------------------------
                action_reward_storage.append_batch(
                    action=stack_numpy_hierarchies(*actions),
                    reward=reward,
                )
                
                if store_activations:
                    #a = {
                    #    key:(value.cpu().numpy().squeeze(axis=0)
                    #        if value.shape[0] == 1 else value.cpu().numpy())
                    #    for key, value in x.items()
                    #}
                    a = interface.numpy_activations(x)
                    activation_storage.append_batch(activations=a)
                
                update = action_reward_storage.num_finished_seqs() - progress.n
                progress.update(update)
    
    if store_observations:
        episodes = observation_storage | action_reward_storage
    else:
        episodes = action_reward_storage
    
    if store_activations:
        episodes = episodes | activation_storage
    
    return episodes

#def test_epoch(config, epoch, test_env, model, interface, log, clock):
def test_episodes(config, epoch, episodes, interface, test_log):
    print('-'*80)
    print('Testing')
    
    avg_terminal_reward = 0.
    reward_bins = [0 for _ in range(11)]
    for seq_id in episodes.finished_seqs:
        seq = episodes.get_seq(seq_id)
        reward = seq['reward'][-1]
        avg_terminal_reward += reward
        reward_bin = int(reward * 10)
        reward_bins[reward_bin] += 1
    
    n = episodes.num_finished_seqs()
    if n:
        avg_terminal_reward /= n
    
    print('Average Terminal Reward: %f'%avg_terminal_reward)
    test_log.log(terminal_reward=avg_terminal_reward)
    
    if n:
        bin_percent = [b/n for b in reward_bins]
        for i, (p, c) in enumerate(zip(bin_percent, reward_bins)):
            low = i * 0.1
            print('%.01f'%low)
            print('|' * round(p * 40) + ' (%i)'%c)
    
    #avg_reward = 0.
    #entries = 0
    #for seq_id in range(episodes.num_seqs()):
    #    seq = episodes.get_seq(seq_id)
    #    avg_reward += numpy.sum(seq['reward'])
    #    entries += seq['reward'].shape[0]
    #
    #avg_reward /= entries
    #print('Average Reward: %f'%avg_reward)
    #log.add_scalar('val/reward', avg_reward, clock[0])
    
    if hasattr(interface, 'test_episodes'):
        interface.test_episodes(episodes, test_log)
    
    #return episodes

def visualize_episodes(config, epoch, episodes, interface):
    #frequency = config.visualization_frequency
    #if frequency and epoch % frequency == 0:
    print('-'*80)
    print('Generating Visualizations')
    
    visualization_directory = os.path.join(
        './visualization',
        #os.path.split(log.log_dir)[-1],
        'epoch_%04i'%epoch,
    )
    if not os.path.exists(visualization_directory):
        os.makedirs(visualization_directory)
    
    interface.visualize_episodes(epoch, episodes, visualization_directory)

def save_checkpoint(
    config, epoch, model, optimizer, scheduler, train_log, test_log):
    #checkpoint_directory = os.path.join(
    #    './checkpoint', os.path.split(log.log_dir)[-1])
    
    if not os.path.exists(config.checkpoint_directory):
        os.makedirs(config.checkpoint_directory)
    
    path = os.path.join(
        config.checkpoint_directory, 'checkpoint_%04i.pt'%epoch)
    print('-'*80)
    print('Saving checkpoint to: %s'%path)
    checkpoint = {
        'epoch' : epoch,
        'config' : config.as_dict(),
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'train_log' : train_log.get_state(),
        'test_log' : test_log.get_state(),
    }
    torch.save(checkpoint, path)
