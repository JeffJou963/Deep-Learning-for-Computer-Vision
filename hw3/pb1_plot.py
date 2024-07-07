import os
import clip
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torchvision.transforms as transforms

from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader

from dataloader import pb1_dataset

BATCH_SIZE = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
# parser.add_argument('--img_dir', default = './hw3_data/p1_data/val')
# parser.add_argument('--label_dir', default = './hw3_data/p1_data')
parser.add_argument('--img_dir', default = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p1_data/val')
parser.add_argument('--label_dir', default = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p1_data')
parser.add_argument('--output_dir', default = './output_p1/pred.csv')
args = parser.parse_args()

# os.makedirs(args.output_dir, exist_ok=True)
model, preprocess = clip.load('ViT-B/32', device=device)

dataset = pb1_dataset(args.img_dir, preprocess)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

with open(f'{args.label_dir}/id2label.json', 'r') as f:
    id2label = json.load(f)
label_text = list(id2label.values())

text = torch.cat([clip.tokenize(f"This is a photo of a {label}") for label in label_text]).to(device)

data_iter = iter(loader)
plt.figure(figsize=(10,8))

for idx in range(3):
    image, img_class, img_name = next(data_iter)
    image = image.to(device)

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)

    similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
    probs, indices = similarity[0].topk(5)
    print(probs)
    print(indices)

    plt.subplot(3,2, 2*idx+1)
    image = image.squeeze(0)
    image = np.transpose(image.cpu().numpy(),(1,2,0))
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(3,2, 2*idx+2)
    y = range(len(probs))
    plt.barh(y, [p.detach().cpu().numpy() for p in probs])

    plt.yticks(y, [f'This is a photo of a {label_text[indices[i]]}' for i in range(5)])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'./plot_pb1a.png')
