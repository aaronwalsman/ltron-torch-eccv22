import random

import numpy

import torch
from torch.distributions import Categorical

from ltron.compression import batch_deduplicate_tiled_seqs
from ltron.dataset.paths import get_dataset_info
from ltron.hierarchy import index_hierarchy
from ltron.gym.envs.reassembly_env import reassembly_template_action

from ltron_torch.models.padding import cat_padded_seqs, make_padding_mask
from ltron_torch.gym_tensor import (
    gym_space_to_tensors, default_tile_transform, default_image_transform)
from ltron_torch.train.optimizer import OptimizerConfig, adamw_optimizer
from ltron_torch.models.compressed_transformer import (
    CompressedTransformer, CompressedTransformerConfig)
from ltron_torch.models.sequence_fcn import (
    named_resnet_independent_sequence_fcn)
from ltron_torch.models.heads import (
    LinearMultiheadDecoder, Conv2dMultiheadDecoder)


# build functions ==============================================================

def explicit_decoder(config):
    return ('{'
        # viewpoint
        f'"workspace_viewpoint":8,'
        f'"handspace_viewpoint":8,'
        
        # cursor
        f'"workspace_cursor_activate":2,'
        f'"workspace_cursor_y":{config.workspace_map_height},'
        f'"workspace_cursor_x":{config.workspace_map_width},'
        f'"workspace_cursor_p":2,'
        f'"handspace_cursor_activate":2,'
        f'"handspace_cursor_y":{config.handspace_map_height},'
        f'"handspace_cursor_x":{config.handspace_map_width},'
        f'"handspace_cursor_p":2,'
        
        # insert brick
        f'"insert_brick_class":{config.num_classes},'
        f'"insert_brick_color":{config.num_colors},'
        
        # disassembly
        f'"disassembly":2,'
        
        # pick and place
        f'"pick_and_place":3,'
        
        # rotate
        f'"rotate":4,'
        
        # reassembly
        f'"reassembly":3'
        '}'
    )
    

def build_reassembly_model(config):
    print('-'*80)
    print('Building reassembly model')
    wh = config.workspace_image_height // config.tile_height
    ww = config.workspace_image_width  // config.tile_width
    hh = config.handspace_image_height // config.tile_height
    hw = config.handspace_image_width  // config.tile_width
    model_config = CompressedTransformerConfig(
        data_shape = (2, config.max_episode_length, wh, ww, hh, hw),
        causal_dim=1,
        include_tile_embedding=True,
        include_token_embedding=True,
        tile_h=config.tile_height,
        tile_w=config.tile_width,
        token_vocab=2,

        num_blocks=8, # who knows?

        decode_input=False,
        decoder_tokens=1,
        decoder_channels=explicit_decoder(config)
    )
    return CompressedTransformer(model_config).cuda()


def build_resnet_disassembly_model(config):
    print('-'*80)
    print('Building resnet disassembly model')
    global_heads = 9
    dense_heads = 3
    model = named_resnet_independent_sequence_fcn(
        'resnet50',
        256,
        global_heads = LinearMultiheadDecoder(2048, global_heads),
        dense_heads = Conv2dMultiheadDecoder(256, dense_heads, kernel_size=1)
    )
    
    return model.cuda()


# input and output utilities ===================================================

def observations_to_resnet_tensors(train_config, observation, pad):
    
    frames = observation['workspace_color_render']
    s, b, h, w, c = frames.shape
    frames = frames.reshape(s*b, h, w, c)
    frames = [default_image_transform(frame) for frame in frames]
    frames = torch.stack(frames)
    frames = frames.view(s, b, c, h, w)
    
    return frames

def observations_to_tensors(train_config, observation, pad):

    wh = train_config.workspace_image_height
    ww = train_config.workspace_image_width
    hh = train_config.handspace_image_height
    hw = train_config.handspace_image_width
    th = train_config.tile_height
    tw = train_config.tile_width

    # make tiles ---------------------------------------------------------------
    wx, wi, w_pad = batch_deduplicate_tiled_seqs(
        observation['workspace_color_render'], pad, tw, th,
        background=102,
    )
    wi = numpy.insert(wi, (0,3,3), -1, axis=-1)
    wi[:,:,0] = 0
    b = wx.shape[1]
    
    #if wx.shape[0] > 1000:
    #    from splendor.image import save_image
    #    for i in range(observation['workspace_color_render'].shape[0]):
    #        save_image(
    #            observation['workspace_color_render'][i,0], './tmp_%04i.png'%i)
    
    hx, hi, h_pad = batch_deduplicate_tiled_seqs(
        observation['handspace_color_render'], pad, tw, th,
        background=102,
    )
    hi = numpy.insert(hi, (0,1,1), -1, axis=-1)
    hi[:,:,0] = 0

    # move tiles to torch/cuda -------------------------------------------------
    wx = torch.FloatTensor(wx)
    hx = torch.FloatTensor(hx)
    w_pad = torch.LongTensor(w_pad)
    h_pad = torch.LongTensor(h_pad)
    tile_x, tile_pad = cat_padded_seqs(wx, hx, w_pad, h_pad)
    tile_x = default_tile_transform(tile_x).cuda()
    tile_pad = tile_pad.cuda()
    tile_i, _ = cat_padded_seqs(
        torch.LongTensor(wi), torch.LongTensor(hi), w_pad, h_pad)
    tile_i = tile_i.cuda()

    # make tokens --------------------------------------------------------------
    #batch_len = len_hierarchy(batch)
    batch_len = numpy.max(pad)
    token_x = torch.LongTensor(
        observation['reassembly']['reassembling']).cuda()
    token_i = torch.ones((batch_len,b,6), dtype=torch.long) * -1
    token_i[:,:,0] = 0
    token_i[:,:,1] = torch.arange(batch_len).unsqueeze(-1)
    token_i = token_i.cuda()
    token_pad = torch.LongTensor(pad).cuda()

    # make decoder indices and pad ---------------------------------------------
    decoder_i = (
        torch.arange(batch_len).unsqueeze(-1).expand(batch_len, b))
    decoder_pad = torch.LongTensor(pad).cuda()

    return (
        tile_x, tile_i, tile_pad,
        token_x, token_i, token_pad,
        decoder_i, decoder_pad,
    )


def sample_or_max(logits, mode):
    if mode == 'sample':
        distribution = Categorical(logits=logits)
        return distribution.sample()
    elif mode == 'max':
        return torch.argmax(logits, dim=-1)
    else:
        raise NotImplementedError


def logits_to_actions(logits, num_classes, num_colors, mode='sample'):
    
    s, b = logits['disassembly'].shape[:2]
    
    #action_mode = sample_or_max(logits['mode'].view(-1,8), mode).cpu().numpy()
    
    workspace_viewpoint = sample_or_max(
        logits['workspace_viewpoint'].view(-1,8), mode).cpu().numpy()
    
    handspace_viewpoint = sample_or_max(
        logits['handspace_viewpoint'].view(-1,8), mode).cpu().numpy()
    
    cursor = {}
    for space in ('workspace', 'handspace'):
        cursor[space] = {}
        for yxpa in 'y', 'x', 'p', 'activate':
            logit_name = '%s_cursor_%s'%(space, yxpa)
            d = logits[logit_name].shape[-1]
            cursor[space][yxpa] = sample_or_max(
                logits[logit_name].view(-1, d), mode).cpu().numpy()
    
    disassembly = sample_or_max(
        logits['disassembly'].view(-1,2), mode).cpu().numpy()
    
    insert_class = sample_or_max(
        logits['insert_brick_class'].view(-1, num_classes), mode).cpu().numpy()
    
    insert_color = sample_or_max(
        logits['insert_brick_color'].view(-1, num_colors), mode).cpu().numpy()
    
    pick_and_place = sample_or_max(
        logits['pick_and_place'].view(-1,3), mode).cpu().numpy()
    
    rotate = sample_or_max(
        logits['rotate'].view(-1,4), mode).cpu().numpy()
    
    reassembly = sample_or_max(
        logits['reassembly'].view(-1,3), mode).cpu().numpy()
    
    # assemble actions
    actions = []
    for i in range(b):
        action = reassembly_template_action()
        action['workspace_viewpoint'] = workspace_viewpoint[i]
        action['handspace_viewpoint'] = handspace_viewpoint[i]
        
        for space in 'workspace', 'handspace':
            component_name = '%s_cursor'%space
            action[component_name] = {
                'activate' : cursor[space]['activate'][i],
                'position' : numpy.array([
                    cursor[space]['y'][i], cursor[space]['x'][i]]),
                'polarity' : cursor[space]['p']
            }
        
        action['disassembly'] = disassembly[i]
        
        action['insert_brick'] = {
            'class_id' : insert_class[i],
            'color_id' : insert_color[i],
        }
        action['pick_and_place'] = pick_and_place[i]
        action['rotate'] = rotate[i]
        action['reassembly'] = reassembly[i]
        
        actions.append(action)
    
    return actions
