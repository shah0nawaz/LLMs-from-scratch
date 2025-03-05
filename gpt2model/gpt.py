import torch
import torch.nn as nn
from .blocks import TransformerBlock
from .layers import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dropout = nn.Dropout(cfg['drop_rate'])
        self.final_norm = LayerNorm(cfg)
        self.final_linear_layer = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) 
             for _ in range(cfg['n_layers'])]
        )
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        token_embeddings = self.tok_emb(input_ids)
        positional_embeddings = self.pos_emb(
            torch.arange(seq_len, device=input_ids.device))
        
        input_embeddings = token_embeddings + positional_embeddings
        x = self.dropout(input_embeddings)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.final_linear_layer(x)
        return logits
