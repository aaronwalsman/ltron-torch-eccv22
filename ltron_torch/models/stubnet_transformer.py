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
from ltron_torch.models.stubnet_decoder import (
    StubnetDecoderConfig, StubnetDecoder)
from ltron_torch.models.padding import decat_padded_seq

class StubnetTransformerConfig(
    HandTableEmbeddingConfig,
    TransformerConfig,
    StubnetDecoderConfig
):
    encoder_blocks = 12
    encoder_channels = 768
    encoder_residual_channels = None
    encoder_heads = 12
    
    init_weights = True

class StubnetTransformer(Module):
    def __init__(self, config, checkpoint=None):
        super().__init__()
        self.config = config
        
        # build the token embedding
        config.token_vocabulary = 3
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
        
        # build the decoder
        self.decoder = StubnetDecoder(config)
        
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
    
    def zero_all_memory(self):
        self.encoder.zero_all_memory()
    
    def forward(self,
        table_image, hand_image,
        table_tiles, table_t, table_yx, table_pad,
        hand_tiles, hand_t, hand_yx, hand_pad,
        token_x,
        table_cursor_yx,
        table_cursor_p,
        hand_cursor_yx,
        hand_cursor_p,
        token_t, token_pad,
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
        
        # extract decoder tokens
        tile_pad = table_pad + hand_pad
        tile_encode_x, token_encode_x = decat_padded_seq(x, tile_pad, token_pad)
        decode_x = token_encode_x[1::2]
        
        # need to select out the necessary locations using
        # table_cursor_activate and hand_cursor_activate
        # then modify observations_to_tensors to only pass in
        # table_image and hand_image for the locations where the cursors are
        # being decoded
        #s, b, c = decode_x.shape
        #hand_table_decode_x = decode_x.reshape(s*b, c)
        #table_decode_x = hand_table_decode_x[table_cursor_activate.view(-1)]
        #hand_decode_x = hand_table_decode_x[hand_cursor_activate.view(-1)]
        
        # use the decoder to decode
        x = self.decoder(
            decode_x,
            table_cursor_activate,
            table_image,
            hand_cursor_activate,
            hand_image,
            insert_activate,
        )
        
        return x
