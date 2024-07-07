import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding (1,60,768)
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class Cross_Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.i_attn = nn.Linear(cfg.n_embd, 2*cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x, encoder_output):
        # batch, context, embedding
        B, T, C = x.size() # (b,max_len,768)
        q = self.c_attn(x) # (b,max_len,768)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (b,12,max_len,64)
        
        B, I, C = encoder_output.size()              # (b,60,768)
        k, v = self.i_attn(encoder_output).split(self.n_embd, dim=2)    # (b,257,768)
        k = k.view(B, I, self.n_head, C // self.n_head).transpose(1, 2) # (b,12,257,64)
        v = v.view(B, I, self.n_head, C // self.n_head).transpose(1, 2)
        # q (1,12,257,64), kT (1,12,64,60)
        # q@kT = att (1,12,257,60)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)  #(1,12, 257,257)
        # att@v.T = (1,257,12,64)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C)) #(1,257,768)

class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ln_3 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.cross_attn = Cross_Attention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))

    def forward(self, x, encoder_output):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), encoder_output)
        x = x + self.mlp(self.ln_3(x))
        return x

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.linear = nn.Linear(1408, cfg.n_embd)  # (1,257,1408->768)

        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, encoder_output, caption_in):
        encoder_output = self.linear(encoder_output)
        x = torch.narrow(caption_in, 1, 0, min(caption_in.size(1), self.block_size))    #(1,max_cap_len)
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0) #(1,max_cap_len)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)             #(1,max_cap_len,768)
        # combine x and encoder_output
        for layer in self.transformer.h:
            x = layer(x, encoder_output)
        # x (1,257,768)
        x = self.lm_head(self.transformer.ln_f(x)) # x (1,max_cap_len,50257)
        return x
