import torch
from torch.nn import Module

from ltron.config import Config

from ltron_torch.models.padding import cat_padded_seqs
from ltron_torch.models.positional_encoding import LearnedPositionalEncoding
from ltron_torch.models.embedding import TileEmbedding, TokenEmbedding
from ltron_torch.models.heads import LinearMultiheadDecoder

class HandTableEmbeddingConfig(Config):
    tile_h = 16
    tile_w = 16
    tile_c = 3
    table_h = 256
    table_w = 256
    hand_h = 96
    hand_w = 96
    
    max_sequence_length = 1024
    
    embedding_dropout = 0.1
    
    token_vocabulary = 2
    
    def set_dependents(self):
        assert self.table_h % self.tile_h == 0
        assert self.table_w % self.tile_w == 0
        self.table_tiles_h = self.table_h // self.tile_h
        self.table_tiles_w = self.table_w // self.tile_w
        self.table_tiles = self.table_tiles_h * self.table_tiles_w
        
        assert self.hand_h % self.tile_h == 0
        assert self.hand_w % self.tile_w == 0
        self.hand_tiles_h = self.hand_h // self.tile_h
        self.hand_tiles_w = self.hand_w // self.tile_w
        self.hand_tiles = self.hand_tiles_h * self.hand_tiles_w
        
        self.spatial_tiles = self.table_tiles + self.hand_tiles

class HandTableEmbedding(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # build the tokenizers
        self.tile_embedding = TileEmbedding(
            config.tile_h,
            config.tile_w,
            config.tile_c,
            config.encoder_channels,
            config.embedding_dropout,
        )
        self.token_embedding = TokenEmbedding(
            config.token_vocabulary,
            config.encoder_channels,
            config.embedding_dropout,
        )
        
        # let's give this another look soon
        if self.config.factor_cursor_distribution:
            self.table_cursor_embedding = TokenEmbedding(
                config.table_decoder_pixels,
                config.encoder_channels,
                config.embedding_dropout,
            )
            self.table_polarity_embedding = TokenEmbedding(
                2, config.encoder_channels, config.embedding_dropout)
            
            self.hand_cursor_embedding = TokenEmbedding(
                config.hand_decoder_pixels,
                config.encoder_channels,
                config.embedding_dropout,
            )
            self.hand_polarity_embedding = TokenEmbedding(
                2, config.encoder_channels, config.embedding_dropout)
        
        # build the positional encodings
        self.spatial_position_encoding = LearnedPositionalEncoding(
            config.encoder_channels, config.spatial_tiles)
        self.temporal_position_encoding = LearnedPositionalEncoding(
            config.encoder_channels, config.max_sequence_length)
    
    def forward(self,
        table_tiles, table_t, table_yx, table_pad,
        hand_tiles, hand_t, hand_yx, hand_pad,
        token_x,
        table_cursor_yx,
        table_cursor_p,
        hand_cursor_yx,
        hand_cursor_p,
        token_t, token_pad,
    ):
        
        # linearize table_yx and hand_yx
        table_w = self.config.table_tiles_w
        table_yx = table_yx[...,0] * table_w + table_yx[...,1]
        hand_w = self.config.hand_tiles_w
        hand_yx = hand_yx[...,0] * hand_w + hand_yx[...,1]
        
        # cat table and hand tiles
        tile_x, tile_pad = cat_padded_seqs(
            table_tiles, hand_tiles, table_pad, hand_pad)
        tile_t, _ = cat_padded_seqs(table_t, hand_t, table_pad, hand_pad)
        tile_yx, _ = cat_padded_seqs(table_yx, hand_yx, table_pad, hand_pad)
        
        # make the tile embeddings
        tile_x = self.tile_embedding(tile_x)
        tile_pt = self.temporal_position_encoding(tile_t)
        tile_pyx = self.spatial_position_encoding(tile_yx)
        tile_x = tile_x + tile_pt + tile_pyx
        
        # make the tokens
        token_x = self.token_embedding(token_x)
        token_pt = self.temporal_position_encoding(token_t)
        token_x = token_x + token_pt
        
        if self.config.factor_cursor_distribution:
            table_cursor_yx = self.table_cursor_embedding(table_cursor_yx)
            #table_cursor_yx = table_cursor_yx + token_pt
            table_cursor_p = self.table_polarity_embedding(table_cursor_p)
            #table_cursor_p = table_cursor_p + token_pt
            # THIS IS ALL SO GROSS
            if table_cursor_yx.shape[0] == token_pt.shape[0]//2:
                table_pt = token_pt[::2]
                table_t = token_t[::2]
                table_pad = (token_pad/2).long()
            else:
                table_pt = token_pt
                table_t = token_t
                table_pad = token_pad
            table_x = table_cursor_yx + table_cursor_p + table_pt
            
            hand_cursor_yx = self.hand_cursor_embedding(hand_cursor_yx)
            #hand_cursor_yx = hand_cursor_yx + token_pt
            hand_cursor_p = self.hand_polarity_embedding(hand_cursor_p)
            #hand_cursor_p = hand_cursor_p + token_pt
            if hand_cursor_yx.shape[0] == token_pt.shape[0]//2:
                hand_pt = token_pt[::2]
                hand_t = token_t[::2]
                hand_pad = (token_pad/2).long()
            else:
                hand_pt = token_pt
                hand_t = token_t
                hand_pad = token_pad
            hand_x = hand_cursor_yx + hand_cursor_p + hand_pt
            
            # all these cat_padded_seqs could probably be done more efficiently
            # in a single function that rolls them all together at once
            #table_x, table_pad = cat_padded_seqs(
            #    table_cursor_yx, table_cursor_p, token_pad, token_pad)
            #table_t, _ = cat_padded_seqs(
            #    token_t, token_t, token_pad, token_pad)
            #hand_x, hand_pad = cat_padded_seqs(
            #    hand_cursor_yx, hand_cursor_p, token_pad, token_pad)
            #hand_t, _ = cat_padded_seqs(
            #    token_t, token_t, token_pad, token_pad)
            
            cursor_x, cursor_pad = cat_padded_seqs(
                table_x, hand_x, table_pad, hand_pad)
            cursor_t, _ = cat_padded_seqs(
                table_t, hand_t, table_pad, hand_pad)
            token_x, new_token_pad = cat_padded_seqs(
                token_x, cursor_x, token_pad, cursor_pad)
            token_t, _ = cat_padded_seqs(
                token_t, cursor_t, token_pad, cursor_pad)
            token_pad = new_token_pad
        
        # concatenate the tile and discrete tokens
        x, pad = cat_padded_seqs(tile_x, token_x, tile_pad, token_pad)
        t, _ = cat_padded_seqs(tile_t, token_t, tile_pad, token_pad)
        
        return x, t, pad
