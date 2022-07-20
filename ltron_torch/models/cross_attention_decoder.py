import torch
from torch.nn import Module, Linear, LayerNorm, Sequential

from ltron_torch.models.mask import padded_causal_mask
from ltron_torch.models.positional_encoding import LearnedPositionalEncoding
from ltron_torch.models.transformer import (
    TransformerConfig,
    Transformer,
    TransformerBlock,
)
from ltron_torch.models.heads import LinearMultiheadDecoder
from ltron_torch.models.hand_table_embedding import (
    HandTableEmbeddingConfig, HandTableEmbedding)
from ltron_torch.models.padding import get_seq_batch_indices

class CrossAttentionDecoderConfig(TransformerConfig):
    table_decode_tokens_h = 16
    table_decode_tokens_w = 16
    table_decode_h = 64
    table_decode_w = 64
    hand_decode_tokens_h = 6
    hand_decode_tokens_w = 6
    hand_decode_h = 24
    hand_decode_w = 24
    
    global_tokens = 1
    
    max_sequence_length = 1024
    
    blocks = 4
    channels = 768
    residual_channels = None
    heads = 12
    
    cursor_channels = 2
    num_modes = 20
    num_shapes = 6
    num_colors = 6
    
    old_style = False
    
    def set_dependents(self):
        assert self.table_decode_h % self.table_decode_tokens_h == 0, (
            '%i, %i'%(self.table_decode_tokens_h, self.table_decode_h))
        assert self.table_decode_w % self.table_decode_tokens_w == 0
        self.upsample_h = self.table_decode_h // self.table_decode_tokens_h
        self.upsample_w = self.table_decode_w // self.table_decode_tokens_w
        self.table_decode_tokens = (
            self.table_decode_tokens_h * self.table_decode_tokens_w)
        
        assert self.hand_decode_h % self.hand_decode_tokens_h == 0
        assert self.hand_decode_w % self.hand_decode_tokens_w == 0
        assert self.hand_decode_h//self.hand_decode_tokens_h == self.upsample_h
        assert self.hand_decode_w//self.hand_decode_tokens_w == self.upsample_w
        self.hand_decode_tokens = (
            self.hand_decode_tokens_h * self.hand_decode_tokens_w)
        
        self.decode_tokens = self.table_decode_tokens + self.hand_decode_tokens
        
        self.table_decoder_pixels = self.table_decode_h * self.table_decode_w
        self.hand_decoder_pixels = self.hand_decode_h * self.hand_decode_w
        self.decoder_pixels = (
            self.table_decoder_pixels + self.hand_decoder_pixels)

class CrossAttentionDecoder(Module):
    def __init__(self, config):
        super().__init__()
        
        # store config
        self.config = config
        
        # build the positional encodings
        if self.config.old_style:
            self.spatial_position_encoding = LearnedPositionalEncoding(
                config.channels,
                config.decode_tokens + config.global_tokens,
            )
        else:
            self.table_position_encoding = LearnedPositionalEncoding(
                config.channels,
                16*16,
            )
            self.hand_position_encoding = LearnedPositionalEncoding(
                config.channels,
                6*6,
            )
            self.global_position_encoding = LearnedPositionalEncoding(
                config.channels,
                1,
            )
        
        self.temporal_position_encoding = LearnedPositionalEncoding(
            config.channels, config.max_sequence_length)
        
        # build the cross-attention block
        self.block = TransformerBlock(config)
        
        # output
        self.norm = LayerNorm(config.channels)
        
        upsample_channels = (
            config.cursor_channels * config.upsample_h * config.upsample_w)
        self.table_decoder = Linear(
            config.channels, upsample_channels)
        self.hand_decoder = Linear(
            config.channels, upsample_channels)
        
        if self.config.old_style:
            #self.mode_decoder = Linear(
            #    config.channels, config.global_channels)
            global_head_spec = {
                'mode' : config.num_modes,
                'shape' : config.num_shapes,
                'color' : config.num_colors,
            }
            self.global_decoder = LinearMultiheadDecoder(
                config.channels,
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
                config.channels,
                mode_head_spec,
            )
            self.insert_decoder = LinearMultiheadDecoder(
                config.channels,
                insert_head_spec,
            )
    
    def forward(self,
        tq,
        pad_q,
        xk,
        tk,
        pad_k,
        table_cursor_activate,
        hand_cursor_activate,
        insert_activate,
        use_memory=None
    ):
        uh = self.config.upsample_h
        uw = self.config.upsample_w
        uc = self.config.cursor_channels
        
        # use the positional encoding to generate the query tokens
        if self.config.old_style:
            x_spatial = self.spatial_position_encoding.encoding
        else:
            x_table = self.table_position_encoding.encoding
            x_hand = self.hand_position_encoding.encoding
            x_global = self.global_position_encoding.encoding
        
        x_temporal = self.temporal_position_encoding(tq)
        s, b, c = x_temporal.shape
        #table_hw, c = x_table.shape
        #table_xq = x_temporal.view(s, 1, b, c) + x_table.view(1, table_hw, 1, c)
        ###xq = xq.view(s*hw, b, c)
        #table_s, table_b = torch.where(table_cursor_activate)
        #hand_s, hand_b = torch.where(hand_cursor_activate)
        #table_xq = xq[table_s,:,table_b]
        #hand_xq = xq[hand_s,:,hand_b]
        # OK, first pass of including activations doesn't work.
        # I can't simply merge sparse s and b dimensions into one long dimension
        # because I need the batch dimension to remain so that the transformer
        # can keep the different batch entries from talking to each other.
        # So I need to go back and try again using padding.  Not a huge deal,
        # but a little more complicated than what I tried above.
        
        # take 2: find the activation locations
        #spatial_cursor_activate = table_cursor_activate | hand_cursor_activate
        def compute_attention(region_cursor_activate, x_region):
            hw, c = x_region.shape
            xq = x_temporal.view(s, 1, b, c) + x_region.view(1, hw, 1, c)
            
            region_b, region_s = torch.where(region_cursor_activate.T)
            region_pad = torch.sum(region_cursor_activate, dim=0)
            region_len = torch.max(region_pad)
            region_pad_s, region_pad_b = get_seq_batch_indices(region_pad)
            # still not sure I've verified this enough
            
            # take 2: region_xq
            region_xq = torch.zeros(region_len, hw, b, c, device=xq.device)
            region_xq_unrolled = xq[region_s, :, region_b]
            region_xq[region_pad_s, :, region_pad_b] = region_xq_unrolled
            region_xq = region_xq.view(region_len * hw, b, c)
            
            # take 2: region_tq
            region_tq = torch.zeros(
                region_len, b, dtype=torch.long, device=tq.device)
            region_tq_unrolled = tq[region_s, region_b]
            region_tq[region_pad_s, region_pad_b] = region_tq_unrolled
            region_tq = region_tq.view(
                region_len, 1, b).expand(region_len, hw, b)
            region_tq = region_tq.reshape(region_len * hw, b)
            
            ###tq = tq.view(s, 1, b).expand(s, hw, b).reshape(s*hw, b)
            ###pad_q = pad_q * hw
            #region_tq = tq[region_s, region_b]
            #hand_tq = tq[hand_s, hand_b]
            
            # compute the mask
            ###mask = padded_causal_mask(tq, pad_q, tk, pad_k)
            region_mask_pad = region_pad * hw
            region_mask = padded_causal_mask(
                region_tq, region_mask_pad, tk, pad_k)
            
            # use the transformer block to compute the output
            ###x = self.block(
            ###    xq, pad_k, xk=xk, mask=mask, use_memory=use_memory)
            region_x = self.block(
                region_xq,
                pad_k,
                xk=xk,
                mask=region_mask,
                use_memory=use_memory
            )
        
            # reshape the output
            uhwc = uh*uw*uc
            region_x = region_x.view(region_len, hw, b, c)
            
            return region_x, region_pad_s, region_pad_b
        
        if self.config.old_style:
            hw, c = x_spatial.shape
            xq = x_temporal.view(s, 1, b, c) + x_spatial.view(1, hw, 1, c)
            xq = xq.view(s*hw, b, c)
            
            tq = tq.view(s, 1, b).expand(s, hw, b).reshape(s*hw,b)
            pad_q = pad_q * hw
            
            mask = padded_causal_mask(tq, pad_q, tk, pad_k)
            
            x = self.block(xq, pad_k, xk=xk, mask=mask, use_memory=use_memory)
            
            uh = self.config.upsample_h
            uw = self.config.upsample_w
            uc = self.config.cursor_channels
            uhwc = uh*uw*c
            x = x.view(s, hw, b, c)
        
        else:
            table_x, tps, tpb = compute_attention(
                table_cursor_activate, x_table)
            hand_x, hps, hpb = compute_attention(hand_cursor_activate, x_hand)
            global_activate = torch.ones_like(table_cursor_activate)
            global_x, _, _ = compute_attention(global_activate, x_global)
        
        # split off the table tokens, upsample and reshape into a rectangle
        if self.config.old_style:
            table_start = 0
            table_end = table_start + self.config.table_decode_tokens
            table_x = self.table_decoder(x[:,table_start:table_end])
            th = self.config.table_decode_tokens_h
            tw = self.config.table_decode_tokens_w
            table_x = table_x.view(s, th, tw, b, uh, uw, uc)
            table_x = table_x.permute(0, 3, 6, 1, 4, 2, 5)
            table_x = table_x.reshape(s, b, uc, th*uh, tw*uw)
        else:
            table_x = self.table_decoder(table_x)
            th = self.config.table_decode_tokens_h
            tw = self.config.table_decode_tokens_w
            ts = table_x.shape[0]
            table_x = table_x.view(ts, th, tw, b, uh, uw, uc)
            table_x = table_x.permute(0, 3, 6, 1, 4, 2, 5)
            table_x = table_x.reshape(ts, b, uc, th*uh, tw*uw)
            table_x = table_x[tps, tpb]
        
        # split off the hand tokens, upsample and reshape into a rectangle
        if self.config.old_style:
            hand_start = table_end
            hand_end = hand_start + self.config.hand_decode_tokens
            hand_x = self.hand_decoder(x[:,hand_start:hand_end])
            hh = self.config.hand_decode_tokens_h
            hw = self.config.hand_decode_tokens_w
            hand_x = hand_x.view(s, hh, hw, b, uh, uw, uc)
            hand_x = hand_x.permute(0, 3, 6, 1, 4, 2, 5)
            hand_x = hand_x.reshape(s, b, uc, hh*uh, hw*uw)
        else:
            hand_x = self.hand_decoder(hand_x)
            hh = self.config.hand_decode_tokens_h
            hw = self.config.hand_decode_tokens_w
            hs = hand_x.shape[0]
            hand_x = hand_x.view(hs, hh, hw, b, uh, uw, uc)
            hand_x = hand_x.permute(0, 3, 6, 1, 4, 2, 5)
            hand_x = hand_x.reshape(hs, b, uc, hh*uh, hw*uw)
            hand_x = hand_x[hps, hpb]
        
        # split off the global tokens
        if self.config.old_style:
            global_start = hand_end
            global_end = global_start + self.config.global_tokens
        
        assert self.config.global_tokens == 1
        if self.config.old_style:
            #global_x = x[:,global_start:global_end].permute(0, 2, 1, 3)
            #s, b, _, c = global_x.shape
            #global_x = global_x.view(s,b,c)
            global_x = x[:,global_start:global_end]
            s, _, b, c = global_x.shape
            global_x = global_x.view(s,b,c)
        else:
            s, _, b, c = global_x.shape
            global_x = global_x.view(s,b,c)
        
        if self.config.old_style:
            x = self.global_decoder(global_x)
        else:
            x = self.mode_decoder(global_x)
            insert_decode_x = global_x.reshape(s*b,c)[insert_activate.view(-1)]
            x.update(self.insert_decoder(insert_decode_x))
        
        x['table'] = table_x
        x['hand'] = hand_x
        #mode_x = global_x['mode']
        #shape_x = global_x['shape']
        #color_x = global_x['color']
        
        return x
