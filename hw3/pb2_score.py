import os
import json
import math
import timm
import torch
import argparse
import numpy as np
import torch.nn as nn
import loralib as lora
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader

# from dataloader import pb2_dataset
from pb2_model import Transformer
from pb2_model_adapter import Transformer_adapter
from pb2_model_tuning import Transformer_tuning
from pb2_model_lora import Transformer_lora
from tokenizer import BPETokenizer
from p2_evaluate import eval_clip

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/images/val')
parser.add_argument('--out_dir', default='./output_p2/pred.json')
parser.add_argument('--decoder_weights', default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/decoder_model.bin')
args = parser.parse_args()

class pb2_dataset(Dataset):
    def __init__(self, data_dir):
        self.images = []

        self.test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                self.images.append(os.path.join(root, file))
        
    def __getitem__(self, index):
        image = self.test_transform(Image.open(self.images[index]).convert('RGB'))
        img_name = self.images[index].split('/')[-1].split('.')[0]
        return img_name, image

    def __len__(self):
        return len(self.images)

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

encoder_path = 'vit_gigantic_patch14_clip_224'

model = Transformer_lora(encoder_path, args.decoder_weights)
model_path ='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/CIDER0.8410.ckpt'

model.load_state_dict(torch.load(model_path, device), strict=False)
lora.mark_only_lora_as_trainable(model)
for name, param in model.decoder.named_parameters():
    if 'cross_attn' in name or 'ln_2' in name:
        param.requires_grad = True

encoder_file = './encoder.json'
vocab_file = './vocab.bpe'
tokenizer = BPETokenizer(encoder_file, vocab_file)

val_dataset = pb2_dataset(args.data_dir)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

model = model.to(device)

model.eval()
cur_max = ''
cur_min = ''
max_clip = 0
min_clip = 1

with torch.no_grad():
    for val_img_file_name, val_image in tqdm(val_loader):
        json_dict = {}
        val_img_file_name = val_img_file_name[0]

        val_image = val_image.to(device)

        encoder_out = model.encoder.forward_features(val_image)
        total_pred=[]
        cap_in = torch.full((1, 60), 50256)
        for i in range(59):
            cap_in = cap_in.to(device)
            output = model.decoder(encoder_out, cap_in) #(1,60,50257)
            output = torch.argmax(output, dim=-1) # (1,60)
            current = output.squeeze(0)[i].item() # (1)
            cap_in[0][i+1] = current             
            if current == 50256:
                break
            else:
                total_pred.append(current)
        total_pred = tokenizer.decode(total_pred)
        print(f'{val_img_file_name} : {total_pred}')


        json_dict[val_img_file_name] = total_pred

        with open(args.out_dir, 'w') as json_file:
            json.dump(json_dict, json_file, indent=2)
        # cider_score, clip_score = eval(args.out_dir, args.data_dir, val_json_dir)
        clip_score = eval_clip(args.out_dir, args.data_dir)
        print(f'{val_img_file_name}_{clip_score}')

        if clip_score > max_clip:
            max_clip = clip_score
            cur_max = val_img_file_name            
            print('max',max_clip,cur_max)
        if clip_score < min_clip:
            min_clip = clip_score
            cur_min = val_img_file_name            
            print('min',min_clip,cur_min)

    print('MIN',min_clip,cur_min)
    print('MAX',max_clip,cur_max)


    # 000000562675_0.48553466796875

    # 000000034708_0.98388671875