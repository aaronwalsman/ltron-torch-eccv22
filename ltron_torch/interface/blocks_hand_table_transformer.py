import os

import numpy

import torch
from torch.nn.functional import cross_entropy

import tqdm

from ltron.gym.envs.blocks_env import BlocksEnv
from ltron.compression import batch_deduplicate_from_masks

from ltron_torch.gym_tensor import default_tile_transform
from ltron_torch.models.padding import cat_padded_seqs
from ltron_torch.interface.blocks import BlocksInterface

class BlocksHandTableTransformerInterface(BlocksInterface):
    def observation_to_tensors(self, observation, pad):
        # get the device
        device = next(self.model.parameters()).device
        
        # process table tiles
        table_tiles, table_tyx, table_pad = batch_deduplicate_from_masks(
            observation['table_render'],
            observation['table_tile_mask'],
            observation['step'],
            pad,
        )
        
        table_pad = torch.LongTensor(table_pad).to(device)
        table_tiles = default_tile_transform(table_tiles).to(device)
        table_t = torch.LongTensor(table_tyx[...,0]).to(device)
        table_yx = torch.LongTensor(
            table_tyx[...,1] *
            self.model.config.table_tiles_w +
            table_tyx[...,2],
        ).to(device)
        
        # processs hand tiles
        hand_tiles, hand_tyx, hand_pad = batch_deduplicate_from_masks(
            observation['hand_render'],
            observation['hand_tile_mask'],
            observation['step'],
            pad,
        )
        
        hand_pad = torch.LongTensor(hand_pad).to(device)
        hand_tiles = default_tile_transform(hand_tiles).to(device)
        hand_t = torch.LongTensor(hand_tyx[...,0]).to(device)
        hand_yx = torch.LongTensor(
            hand_tyx[...,1] *
            self.model.config.hand_tiles_w +
            hand_tyx[...,2] +
            self.model.config.table_tiles,
        ).to(device)
        
        # cat table and hand ties
        tile_x, tile_pad = cat_padded_seqs(
            table_tiles, hand_tiles, table_pad, hand_pad)
        tile_t, _ = cat_padded_seqs(table_t, hand_t, table_pad, hand_pad)
        tile_yx, _ = cat_padded_seqs(table_yx, hand_yx, table_pad, hand_pad)
        
        # process token x/t/pad
        token_x = torch.LongTensor(observation['phase']).to(device)
        token_t = torch.LongTensor(observation['step']).to(device)
        token_pad = torch.LongTensor(pad).to(device)
        
        # process decode t/pad
        decode_t = token_t
        decode_pad = token_pad
        
        return (
            tile_x, tile_t, tile_yx, tile_pad,
            token_x, token_t, token_pad,
            decode_t, decode_pad,
        )
    
    def rollout_forward(self, terminal, *x):
        use_memory = torch.BoolTensor(~terminal).to(x[0].device)
        return self.model(*x, use_memory=use_memory)
    
    #def loss(self, x, pad, y, log=None, clock=None):
    #    return blocks_loss(self.config, x, pad, y, log, clock)
    
    #def loss(self, x, pad, y, log=None, clock=None):
    #    # split out the components of x
    #    x_table, x_hand, x_mode, x_shape, x_color = x
    #    s,b = x_mode.shape[:2]
    #    
    #    # make the padding mask
    #    pad = torch.LongTensor(pad).to(x_mode.device)
    #    pad_mask = make_padding_mask(pad, (s,b))
    #    
    #    # mode supervision
    #    y_mode = torch.LongTensor(y['mode']).to(x_mode.device)
    #    num_modes = x_mode.shape[-1]
    #    mode_loss = cross_entropy(
    #        x_mode.view(-1,num_modes), y_mode.view(-1), reduction='none')
    #    mode_loss = mode_loss.view(s,b) * ~pad_mask
    #    mode_loss = mode_loss.mean() * self.config.mode_loss_weight
    #    
    #    # table supervision
    #    h, w = x_table.shape[2:4]
    #    table_entries = ((y_mode == 0) | (y_mode == 1)).view(-1)
    #    x_table = x_table.view(s*b, h*w)[table_entries]
    #    y_table_y = torch.LongTensor(y['table_cursor'][:,:,0])
    #    y_table_x = torch.LongTensor(y['table_cursor'][:,:,1])
    #    y_table = (y_table_y * w + y_table_x).view(-1)[table_entries]
    #    y_table = y_table.to(x_table.device)
    #    table_loss = cross_entropy(x_table, y_table)
    #    table_loss = table_loss * self.config.table_loss_weight
    #    
    #    # hand supervision
    #    h, w = x_hand.shape[2:4]
    #    hand_entries = (y_mode == 1).view(-1)
    #    x_hand = x_hand.view(s*b, h*w)[hand_entries]
    #    y_hand_y = torch.LongTensor(y['hand_cursor'][:,:,0])
    #    y_hand_x = torch.LongTensor(y['hand_cursor'][:,:,1])
    #    y_hand = (y_hand_y * w + y_hand_x).view(-1)[hand_entries]
    #    y_hand = y_hand.to(x_hand.device)
    #    hand_loss = cross_entropy(x_hand, y_hand)
    #    hand_loss = hand_loss * self.config.hand_loss_weight
    #    
    #    # shape supervision
    #    shape_entries = (y_mode == 2).view(-1)
    #    num_shapes = x_shape.shape[-1]
    #    x_shape = x_shape.view(s*b, num_shapes)[shape_entries]
    #    y_shape = torch.LongTensor(y['shape']).view(-1)[shape_entries]
    #    y_shape = y_shape.to(x_shape.device)
    #    shape_loss = cross_entropy(x_shape, y_shape)
    #    shape_loss = shape_loss * self.config.shape_loss_weight
    #    
    #    # color supervision
    #    color_entries = (y_mode == 2).view(-1)
    #    num_colors = x_color.shape[-1]
    #    x_color = x_color.view(s*b, num_colors)[color_entries]
    #    y_color = torch.LongTensor(y['color']).view(-1)[color_entries]
    #    y_color = y_color.to(x_color.device)
    #    color_loss = cross_entropy(x_color, y_color)
    #    color_loss = color_loss * self.config.color_loss_weight
    #    
    #    loss = mode_loss + table_loss + hand_loss + shape_loss + color_loss
    #    
    #    if log is not None:
    #        log.add_scalar('train/mode_loss', mode_loss, clock[0])
    #        log.add_scalar('train/table_loss', table_loss, clock[0])
    #        log.add_scalar('train/hand_loss', hand_loss, clock[0])
    #        log.add_scalar('train/shape_loss', shape_loss, clock[0])
    #        log.add_scalar('train/color_loss', color_loss, clock[0])
    #        log.add_scalar('train/total_loss', loss, clock[0])
    #    
    #    return loss
    
    #def tensor_to_actions(self, x, env, mode='sample'):
    #    return blocks_tensor_to_actions(x, mode=mode)
    
    #def tensor_to_actions(self, x, env, mode='sample'):
    #    mode_action = categorical_or_max(x['mode'], mode=mode).cpu().numpy()
    #    shape_action = categorical_or_max(x['shape'], mode=mode).cpu().numpy()
    #    color_action = categorical_or_max(x['color'], mode=mode).cpu().numpy()
    #    
    #    region_yx = []
    #    for region in 'table', 'hand':
    #        s, b, h, w, c = x['region'].shape
    #        x_region = x_region.view(b, 1, h, w)
    #        region_y, region_x = categorical_or_max_2d(x_table, mode=mode)
    #        region_y = region_y.cpu().numpy()
    #        region_x = region_x.cpu().numpy()
    #        region_yx.append((region_y, region_x)
    #    (table_y, table_x), (hand_y, hand_x) = region_yx
    #    
    #    #s, b, h, w, c = x_hand.shape
    #    #x_hand = x_hand.view(b, 1, h, w)
    #    #hand_y, hand_x = categorical_or_max_2d(x_hand, mode=mode)
    #    #hand_y = hand_y.cpu().numpy()
    #    #hand_x = hand_x.cpu().numpy()
    #    
    #    actions = []
    #    for i in range(b):
    #        action = BlocksEnv.no_op_action()
    #        action['mode'] = mode_action[i]
    #        action['shape'] = shape_action[i]
    #        action['color'] = color_action[i]
    #        action['table_cursor'] = numpy.array([table_y[i], table_x[i]])
    #        action['hand_cursor'] = numpy.array([hand_y[i], hand_x[i]])
    #        actions.append(action)
    #    
    #    return actions
    
    #def visualize_episodes(self, epoch, episodes, visualization_directory):
    #    num_seqs = min(
    #        self.config.visualization_seqs, episodes.num_seqs())
    #    for seq_id in tqdm.tqdm(range(num_seqs)):
    #        seq_path = os.path.join(
    #            visualization_directory, 'seq_%06i'%seq_id)
    #        if not os.path.exists(seq_path):
    #            os.makedirs(seq_path)
    #        
    #        seq = episodes.get_seq(seq_id)
    #        seq_len = len_hierarchy(seq)
    #        table_frames = seq['observation']['table_render']
    #        hand_frames = seq['observation']['hand_render']
    #        for frame_id in range(seq_len):
    #            table_frame = table_frames[frame_id]
    #            hand_frame = hand_frames[frame_id]
    #            th, tw = table_frame.shape[:2]
    #            hh, hw = hand_frame.shape[:2]
    #            w = tw + hw
    #            joined_image = numpy.zeros((th, w, 3), dtype=numpy.uint8)
    #            joined_image[:,:tw] = table_frame
    #            joined_image[th-hh:,tw:] = hand_frame
    #            
    #            frame_action = index_hierarchy(seq['action'], frame_id)
    #            frame_mode = int(frame_action['mode'])
    #            frame_shape_id = int(frame_action['shape'])
    #            frame_color_id = int(frame_action['color'])
    #            ty, tx = frame_action['table_cursor']
    #            ty = int(ty)
    #            tx = int(tx)
    #            hy, hx = frame_action['hand_cursor']
    #            hy = int(hy)
    #            hx = int(hx)
    #            
    #            joined_image[ty*4:(ty+1)*4, tx*4:(tx+1)*4] = (0,0,0)
    #            yy = th - hh
    #            joined_image[
    #                yy+hy*4:yy+(hy+1)*4, tw+(hx)*4:tw+(hx+1)*4] = (0,0,0)
    #            
    #            mode_string = 'Mode: %s'%([
    #                'disassemble',
    #                'place',
    #                'pick-up',
    #                'make',
    #                'end',
    #                'no-op'][frame_mode])
    #            shape_string = 'Shape: %s'%str(
    #                self.config.block_shapes[frame_shape_id])
    #            color_string = 'Color: %s'%str(
    #                self.config.block_colors[frame_shape_id])
    #            lines = (mode_string, shape_string, color_string)
    #            joined_image = write_text(joined_image, '\n'.join(lines))
    #            
    #            frame_path = os.path.join(
    #                seq_path,
    #                'frame_%04i_%06i_%04i.png'%(epoch, seq_id, frame_id),
    #            )
    #            save_image(joined_image, frame_path)
