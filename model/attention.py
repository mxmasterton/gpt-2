import math

import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query and value projections for all three heads in a single batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # regularisation
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # not really a "bias", but we follow huggingface naming convention
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))).view(1, 1, config.block_size, config.block_size)
        
    def forward(self, x):
        B, T, C = x.shape # (B,T,C)
        
        qkv = self.c_attn(x) # (B,T,3C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # here we make the heads a batch dimension
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, C)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, C)
        
        # compute the attention scores
        att = (q @ k.tranpose(-2, -1)) / math.sqrt(k.shape[-1]) # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        
        # perform the weighted aggregation of the values
        out = att @ v # (B, nh, T, T) @ (B, nh, T, C) = (B, nh, T, C)
        out = out.tranpose(1, 2).continguous().view(B, T, C) # re-assembling everything
        
        # output projection
        out = self.c_proj(out)
        return out