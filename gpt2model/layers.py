
import torch
import torch.nn as nn

class MaskMultiHeadAttention(nn.Module):
    def __init__(self,d_in, d_out, context_length ,num_heads, dropout=0.5, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = self.d_out // num_heads

        self.W_Q = nn.Linear(d_in, d_out, qkv_bias)
        self.W_K = nn.Linear(d_in, d_out, qkv_bias)
        self.W_V = nn.Linear(d_in, d_out, qkv_bias)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)

        )

        self.projection = nn.Linear(self.d_out, self.d_out)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(b, num_tokens, self.num_heads, self.head_dim)
        K = K.view(b, num_tokens, self.num_heads, self.head_dim)
        V = V.view(b, num_tokens, self.num_heads, self.head_dim)

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        attn_score = Q @ K.transpose(2,3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_score.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
                        attn_score/K.shape[-1]**0.5, dim=-1)
        
        context_vec = self.dropout(attn_weights)
        context_vec = (attn_weights @ V).transpose(1,2)

        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        #print(context_vec)

        out = self.projection(context_vec)

        return out
    

class LayerNorm(nn.Module):
    def __init__(self, cfg, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(cfg['emb_dim']))
        self.shift = nn.Parameter(torch.zeros(cfg['emb_dim']))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        out_norm = (x - mean)/ torch.sqrt(var +  self.eps)
        return self.scale*out_norm + self.shift
    

class GELU(nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, x):

        return 0.8 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))

        ))
    
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg['emb_dim'], 4*cfg['emb_dim'])
        self.gelu = GELU()
        self.fc2 = nn.Linear(4*cfg['emb_dim'], cfg['emb_dim'])

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))
