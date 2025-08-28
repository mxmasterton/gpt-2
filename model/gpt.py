from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from .attention import CausalSelfAttention
from .mlp import MLP

# -------------------------------------------------------------

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x     

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, tokens, targets):
        B, T = tokens.shape
        
        assert T <= self.config.block_size, "Cannot forward sequence of length {}, block size is only {}".format(T, self.config.block_size)
        
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(tokens)
        x = tok_emb + pos_emb
        
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            tokens = tokens.view(B*T)
            
            loss = F.cross_entropy(logits, tokens)
        
        return logits, loss
     
    @classmethod
    def from_pretrained(cls, model_type):
        """ loads pre-trained GPT-2 model weights from huggingface """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: {}".format(model_type))
        
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2":         dict(n_layer=12, n_head=12, n_embd=768),    # 124M parameters
            "gpt2-medium":  dict(n_layer=24, n_head=16, n_embd=1024),   # 350M parameters
            "gpt2-large":   dict(n_layer=36, n_head=20, n_embd=1280),   # 774M parameters
            "gpt2-xl":      dict(n_layer=48, n_head=25, n_embd=1600)    # 1558M parameters
        }[model_type]
        config_args["vocab_size"] = 50257 # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024 # always 1024 for GPT model checkpoints
        
        # create a from-scratch initalised model
        config = GPTConfig(**config_args)
        model = GPT(config)
        
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # discard this mask, not a parameter
        
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        # we have to transpose certain weights when we import them
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        assert len(sd_keys_hf) == len(sd_keys), "mismatched keys: {} != {}".format(len(sd_keys_hf), len(sd_keys))
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model