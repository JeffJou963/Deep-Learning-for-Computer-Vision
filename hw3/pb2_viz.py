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
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# from dataloader import pb2_dataset
from pb2_model import Transformer
from pb2_viz_model import Transformer_lora
from tokenizer import BPETokenizer
from p2_evaluate import eval_clip


class pb2_dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = os.listdir(data_dir)

        self.test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_name = self.image_files[index]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.test_transform(image)
        img_name = img_name.split('.')[0]
        # print(img_name)
        return img_name, image

    def __len__(self):
        return len(self.image_files)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p3_data/images')
parser.add_argument('--out_dir', default='./output_p2')
args = parser.parse_args()

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# encoder_path = 'vit_giant_patch14_clip_224.laion2b'
# encoder_path = 'eva_giant_patch14_224.clip_ft_in1k'
encoder_path = 'vit_gigantic_patch14_clip_224'

decoder_cfg = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/decoder_model.bin'

model = Transformer_lora(encoder_path, decoder_cfg) # 28965888
model_path ='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/CIDER0.8007_17_29_lora_ep4_2.675939.ckpt'
# model_path = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/CIDER0.8410.ckpt'
# model_path = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/CIDER0.8338_6_44_lora_ep9_2.600878.ckpt'
# model_path ='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/ad&lo_30_2_59_lora_val_ep10_2.677285.ckpt'
# model_path = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/vit_29_9_6_lora_val_ep10_3.219812.ckpt'
# model_path = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/vit_29_19_59_lora_val_ep10_2.771368.ckpt'

model.load_state_dict(torch.load(model_path, device), strict=False)
lora.mark_only_lora_as_trainable(model)
for name, param in model.decoder.named_parameters():
    if 'cross_attn' in name or 'ln_2' in name or 'adapter' in name:
        param.requires_grad = True
print('LORALORALORA')

encoder_file = './encoder.json'
vocab_file = './vocab.bpe'
tokenizer = BPETokenizer(encoder_file, vocab_file)

val_json_dir = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/val.json'

val_dataset = pb2_dataset(args.data_dir)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

model = model.to(device)

val_losses = []
json_dict = {}
output_dict = {}
model.eval()
with torch.no_grad():
    for (val_img_file_name,), image in tqdm(val_loader):
        image = image.to(device)

        encoder_out = model.encoder.forward_features(image)
        total_pred=[]
        cap_in = torch.full((1, 60), 50256)
        for i in range(59):
            cap_in = cap_in.to(device)
            output, vizs = model.decoder(encoder_out, cap_in) #(1,60,50257)
            viz = vizs[6]

            output = torch.argmax(output, dim=-1) # (1,60)
            current = output.squeeze(0)[i].item() # (1)
            cap_in[0][i+1] = current             
            if current == 50256:
                break
            else:
                total_pred.append(current)
        total_pred = tokenizer.decode(total_pred)
        print(f'{val_img_file_name} : {total_pred}')

        # score
        json_dir = './output_p2/pred.json'

        json_dict[val_img_file_name] = total_pred
        with open(json_dir, 'w') as json_file:
            json.dump(json_dict, json_file, indent=2)
        clip_score = eval_clip(json_dir, args.data_dir)
        print(f'{val_img_file_name}_{clip_score}')

        # plot
        total_pred = total_pred.split()
        total_pred[-1] = total_pred[-1].split('.')[0] 
        total_pred.append('.')
        total_pred.append('|<endoftext>|')
        total_pred.insert(0,'|<endoftext>|')

        if (len(total_pred) % 2 == 0):
            fig, axs = plt.subplots(2,len(total_pred)//2, figsize=(8,3))
        else:
            fig, axs = plt.subplots(2, len(total_pred)//2+1, figsize=(8,3))
        # print(viz.shape) # [1,12,60,257] -> [1,256] -> [1,16,16]
        viz = torch.mean(viz, dim = 1) # [1, 60,257]

        for i, cur in enumerate(total_pred):
            # print(viz.shape)
            viz_cur = viz[: ,i-1,:]        #[1,257]
            # print(viz_cur.shape)
            viz_cur = viz_cur[:,1:]
            # print(viz_cur.shape)
            viz_sq = viz_cur.view(-1,16,16)

            mask = F.interpolate(viz_sq.unsqueeze(0), size=[224,224], mode="bilinear", align_corners=False)
            mask = mask.squeeze(0).squeeze(0).cpu()
            # print(mask.shape) # [1,1,224,224] -> [224,224]

            if i == 0 :
                axs[0,0].set_title(f'{cur}')
                axs[0,0].imshow(image.squeeze(0).cpu().numpy().transpose(1,2,0))
                axs[0,0].axis('off')

            elif i< len(total_pred)/2:
                axs[0,i].set_title(f'{cur}')
                axs[0,i].imshow(image.squeeze(0).cpu().numpy().transpose(1,2,0))
                axs[0,i].imshow(mask, alpha=0.4, cmap='jet')
                axs[0,i].axis('off')
            else:
                axs[1,i-len(total_pred)//2].set_title(f'{cur}')
                axs[1,i-len(total_pred)//2].imshow(image.squeeze(0).cpu().numpy().transpose(1,2,0))
                axs[1,i-len(total_pred)//2].imshow(mask, alpha=0.4, cmap='jet')
                axs[1,i-len(total_pred)//2].axis('off')

        plt.savefig(f'{args.out_dir}/{val_img_file_name}.png')
        plt.close()





# with open(args.out_dir, 'w') as json_file:
#     json.dump(json_dict, json_file, indent=2)

# cider_score, clip_score = eval(args.out_dir, args.data_dir, val_json_dir)
# print(f"CIDEr: {cider_score} | CLIPScore: {clip_score}")
    
