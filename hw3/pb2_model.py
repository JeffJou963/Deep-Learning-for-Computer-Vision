import math
import collections

import timm
import torch
import argparse
import loralib as lora
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from dataloader import pb2_dataset

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
        att = F.softmax(att, dim=-1)  #(1,12, 257,60)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C)) #(1,257,768)

        # Q: (b,L,H,-1), L=257, H=12, -1=768/12=64
        # K: (b,S,H,-1), S=60
        # V: (b,S,H,-1)

        # att = q@kT: (b,12,257,64)x(b,12,64,60) = (b,12,257,60)
        # return = att@v: (b,12,257,60)x(b,12,60,64) = (b,12,257,64) -> (b,257,12,64)=(b,257,768)

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
        # self.linear = nn.Linear(1024, cfg.n_embd)  # (1,257,1408->768)

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


class Transformer(nn.Module):
    def __init__(self, encoder_path, decoder_cfg_path):
        super().__init__()

        self.encoder = timm.create_model(encoder_path, pretrained=True)
        decoder_cfg = Config(decoder_cfg_path)
        self.decoder = Decoder(decoder_cfg)

    def forward(self, images, caption_in):
        encoder_output = self.encoder.forward_features(images)
        # print(encoder_output.shape)     # (b,257,1408)
        captions = self.decoder(encoder_output, caption_in)

        return captions

if __name__ == '__main__':
    BATCHSIZE = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test')
    parser.add_argument('--data_dir', default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/images/train')
    # parser.add_argument('--data_dir', default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/images/val')
    parser.add_argument('--out_dir', default='./output_p2/pred.json')
    parser.add_argument('--decoder_weights', default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data')
    args = parser.parse_args()

    train_json_dir = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/train.json'
    val_json_dir = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/val.json'
    train_dataset = pb2_dataset(args.data_dir, train_json_dir,'train')
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
    encoder_path = 'eva_giant_patch14_224.clip_ft_in1k'
    decoder_cfg_path = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/decoder_model.bin'
    model = Transformer(encoder_path, decoder_cfg_path)

    iter = iter(train_loader)
    for i in range(3):
        (img_file_name,), image, caption_in, caption_gt = next(iter)
        # caption_in.shape = [1,30]
        # out = model(image, caption_in)
        # print(out.shape) # (1,30,50257)

        encoder_out = model.encoder.forward_features(image)
        print(encoder_out.shape) # (b, 257, 1408)
        decoder_out = model.decoder(encoder_out, caption_in)
        print(decoder_out.shape) # (b, max_cap_len, 50257)

        # x = torch.rand(8,3,224,224)
        # out = model(x)


# x = Embedding()(x)
# print('Emb_out:', x.shape) # (8,197,768)

# # x= Attention()(x)
# # print('MHA_out:', x.shape) # (8,197,768)

# x= Encoder()(x)
# print('Encoder:', x.shape)