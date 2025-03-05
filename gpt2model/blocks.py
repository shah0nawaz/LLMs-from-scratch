import torch
import torch.nn as nn
from .layers import MaskMultiHeadAttention, FeedForward, LayerNorm
class TransformerBlock(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm1 = LayerNorm(cfg)
        self.layer_norm2 = LayerNorm(cfg)

        self.mmhatt = MaskMultiHeadAttention(
                                          d_in = cfg['emb_dim'],
                                          d_out = cfg['emb_dim'],
                                          context_length = cfg['context_length'],
                                          num_heads = cfg['n_heads'],
                                          dropout = cfg['drop_rate'],
                                          qkv_bias = cfg['qkv_bias']
                                         )
        self.ff = FeedForward(cfg)
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.mmhatt(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x