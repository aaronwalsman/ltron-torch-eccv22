import torch
from torch.nn import Module, Linear, LayerNorm, Sequential

from ltron.config import Config

from ltron_torch.models.mask import padded_causal_mask
from ltron_torch.models.positional_encoding import LearnedPositionalEncoding
from ltron_torch.models.heads import LinearMultiheadDecoder
from ltron_torch.models.hand_table_embedding import (
    HandTableEmbeddingConfig, HandTableEmbedding)
from ltron_torch.models.resnet import named_backbone

class StubnetDecoderConfig(Config):
    max_sequence_length = 1024
    
    encoder_channels = 768
    stub_channels = 64
    decode_spatial_locations = 64**2 + 24**2
    
    cursor_channels = 2
    num_modes = 20
    num_shapes = 6
    num_colors = 6
    
    pretrained = True
    
    old_style = False

class StubnetDecoder(Module):
    def __init__(self, config):
        super().__init__()
        
        # store config
        self.config = config
        
        # build the positional encodings
        self.spatial_positional_encoding = LearnedPositionalEncoding(
            config.stub_channels,
            config.decode_spatial_locations,
        )
        
        # build the resnet stub
        self.stubnet = named_backbone(
            'resnet18', 'layer1', pretrained=config.pretrained)
        
        # build the linear layer that converts from encoder to decoder channels
        self.cursor_linear = Linear(
            config.encoder_channels, config.stub_channels*2)
        self.cursor_layer_norm = LayerNorm(config.stub_channels)
        
        if self.config.old_style:
            global_head_spec = {
                'mode' : config.num_modes,
                'shape' : config.num_shapes,
                'color' : config.num_colors,
            }
            self.global_decoder = LinearMultiheadDecoder(
                config.encoder_channels,
                global_head_spec,
            )
        else:
            mode_head_spec = {
                'mode' : config.num_modes,
            }
            insert_head_spec = {
                'shape' : config.num_shapes,
                'color' : config.num_colors,
            }
            self.mode_decoder = LinearMultiheadDecoder(
                config.encoder_channels,
                mode_head_spec,
            )
            self.insert_decoder = LinearMultiheadDecoder(
                config.encoder_channels,
                insert_head_spec,
            )
    
    def forward(
        self,
        decode_x,
        table_cursor_activate,
        table_image,
        hand_cursor_activate,
        hand_image,
        insert_activate,
    ):
        
        # run the stubnet on the input images
        table_x, = self.stubnet(table_image)
        hand_x, = self.stubnet(hand_image)
        
        # add the positional encoding
        p = self.spatial_positional_encoding.encoding
        
        table_p = p[:64**2].T.view(1,-1,64,64)
        table_x = table_x + table_p
        
        hand_p = p[64**2:].T.view(1,-1,24,24)
        hand_x = hand_x + hand_p
        
        # dot the decoder with the hand and table image values
        s, b, _ = decode_x.shape
        hand_table_decode_x = decode_x.reshape(s*b, -1)
        
        table_decode_x = hand_table_decode_x[table_cursor_activate.view(-1)]
        table_cursor_x = self.cursor_linear(table_decode_x)
        sb, cc = table_cursor_x.shape
        c = cc//2
        table_cursor_x = table_cursor_x.view(sb, 2, c)
        table_cursor_x = self.cursor_layer_norm(table_cursor_x)
        table_x = torch.einsum('bchw,bnc->bnhw', table_x, table_cursor_x)
        table_x = table_x / (c**0.5)
        
        hand_decode_x = hand_table_decode_x[hand_cursor_activate.view(-1)]
        hand_cursor_x = self.cursor_linear(hand_decode_x)
        sb, cc = hand_cursor_x.shape
        c = cc//2
        hand_cursor_x = hand_cursor_x.view(sb, 2, c)
        hand_cursor_x = self.cursor_layer_norm(hand_cursor_x)
        hand_x = torch.einsum('bchw,bnc->bnhw', hand_x, hand_cursor_x)
        hand_x = hand_x / (c**0.5)
        
        if self.config.old_style:
            # compute the global output
            x = self.global_decoder(decode_x)
        
        else:
            # compute the node output
            x = self.mode_decoder(decode_x)
            
            # compute the insert output
            insert_decode_x = decode_x.reshape(s*b,-1)[insert_activate.view(-1)]
            x.update(self.insert_decoder(insert_decode_x))
        
        # add table and hand to x
        x['table'] = table_x
        x['hand'] = hand_x
        
        return x
