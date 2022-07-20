import torch
from torch.nn import Module, Linear, LayerNorm, Sequential

from ltron_torch.models.mask import padded_causal_mask
from ltron_torch.models.positional_encoding import LearnedPositionalEncoding
from ltron_torch.models.transformer import (
    TransformerConfig,
    Transformer,
    TransformerBlock,
    init_weights,
)
from ltron_torch.models.heads import LinearMultiheadDecoder
from ltron_torch.models.hand_table_embedding import (
    HandTableEmbeddingConfig, HandTableEmbedding)
from ltron_torch.models.cross_attention_decoder import (
    CrossAttentionDecoderConfig, CrossAttentionDecoder)

class HandTableTransformerConfig(
    HandTableEmbeddingConfig,
    CrossAttentionDecoderConfig,
):
    encoder_blocks = 12
    encoder_channels = 768
    encoder_residual_channels = None
    encoder_heads = 12
    
    decoder_blocks = 4
    decoder_channels = 768
    decoder_residual_channels = None
    decoder_heads = 12
    
    init_weights = True

class HandTableTransformer(Module):
    def __init__(self, config, checkpoint=None):
        super().__init__()
        self.config = config
        
        # build the token embedding
        self.embedding = HandTableEmbedding(config)
        
        # build the transformer
        encoder_config = TransformerConfig.translate(
            config,
            blocks='encoder_blocks',
            channels='encoder_channels',
            residual_channels='encoder_residual_channels',
            num_heads='encoder_heads',
        )
        self.encoder = Transformer(encoder_config)
        
        # build the linear layer that converts from encoder to decoder channels
        self.encode_to_decode = Sequential(
            Linear(config.encoder_channels, config.decoder_channels),
            LayerNorm(config.decoder_channels)
        )
        
        # build the decoder
        decoder_config = CrossAttentionDecoderConfig.translate(
            config,
            blocks='decoder_blocks',
            channels='decoder_channels',
            residual_channels='decoder_residual_channels',
            num_heads='decoder_heads',
        )
        self.decoder = CrossAttentionDecoder(decoder_config)
        
        # initialize weights
        if checkpoint is not None:
            if isinstance(checkpoint, str):
                checkpoint = torch.load(checkpoint)
            if 'token_embedding.embedding.weight' in checkpoint:
                checkpoint['phase_embedding.embedding.weight'] = checkpoint[
                    'token_embedding.embedding.weight']
                del(checkpoint['token_embedding.embedding.weight'])
            if 'mask_embedding.weight' in checkpoint:
                del(checkpoint['mask_embedding.weight'])
            self.load_state_dict(checkpoint)
        elif config.init_weights:
            self.apply(init_weights)
    
    def forward(self,
        #tile_x, tile_t, tile_yx, tile_pad,
        table_tiles, table_t, table_yx, table_pad,
        hand_tiles, hand_t, hand_yx, hand_pad,
        token_x,
        table_cursor_yx,
        table_cursor_p,
        hand_cursor_yx,
        hand_cursor_p,
        token_t, token_pad,
        decode_t, decode_pad,
        table_cursor_activate,
        hand_cursor_activate,
        insert_activate,
        use_memory=None,
    ):
        x, t, pad = self.embedding(
            table_tiles, table_t, table_yx, table_pad,
            hand_tiles, hand_t, hand_yx, hand_pad,
            token_x,
            table_cursor_yx,
            table_cursor_p,
            hand_cursor_yx,
            hand_cursor_p,
            token_t, token_pad,
        )
        
        # use the encoder to encode
        x = self.encoder(x, t, pad, use_memory=use_memory)[-1]
        
        # DIFFERENT
        #print('new x')
        #print(torch.sum(x).cpu())
        
        # convert encoder channels to decoder channels
        x = self.encode_to_decode(x)
        
        #print('new enc to dec x')
        #print(torch.sum(x).cpu())
        
        # use the decoder to decode
        x = self.decoder(
            decode_t,
            decode_pad,
            x,
            t,
            pad,
            table_cursor_activate,
            hand_cursor_activate,
            insert_activate,
            use_memory=use_memory,
        )
        
        #print('new mode/shape/color/table/hand x')
        #print('   ', torch.sum(x['mode']).cpu())
        #print('   ', torch.sum(x['shape']).cpu())
        #print('   ', torch.sum(x['color']).cpu())
        #print('   ', torch.sum(x['table']).cpu())
        #print('   ', torch.sum(x['hand']).cpu())
        
        return x
