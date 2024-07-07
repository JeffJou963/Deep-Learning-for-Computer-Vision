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
# from p2_evaluate import eval

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_path = 'vit_gigantic_patch14_clip_224'

model = Transformer_lora(encoder_path, args.decoder_weights)
model_path ='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/CIDER0.8410.ckpt'

# state_dict = torch.load(model_path)
# print(sum([p.numel() for n,p in state_dict.items()]))

model.load_state_dict(torch.load(model_path, device), strict=False)
lora.mark_only_lora_as_trainable(model)
for name, param in model.decoder.named_parameters():
    if 'cross_attn' in name or 'ln_2' in name:
        param.requires_grad = True

# trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
# save_weights = {k: v for k,v in model.state_dict().items() if k in trainable_weights}
# print('Total params:', sum(p.numel() for p in model.parameters() if p.requires_grad))



encoder_file = './encoder.json'
vocab_file = './vocab.bpe'
tokenizer = BPETokenizer(encoder_file, vocab_file)

val_dataset = pb2_dataset(args.data_dir)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

model = model.to(device)

json_dict = {}
model.eval()
with torch.no_grad():
    for val_img_file_name, val_image in tqdm(val_loader):
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


# val_json_dir = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/val.json'
# cider_score, clip_score = eval(args.out_dir, args.data_dir, val_json_dir)
    
