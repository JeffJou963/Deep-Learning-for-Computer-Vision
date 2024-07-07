import os
import clip
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class pb1_dataset(Dataset):
    def __init__ (self, data_dir, preprocess):
        self.images = []
        self.preprocess = preprocess

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                self.images.append(os.path.join(root, file))

    def __getitem__(self, index):
        image = self.preprocess(Image.open(self.images[index]))
        img_name = self.images[index].split('/')[-1]
        return image, img_name

    def __len__(self):
        return len(self.images)
    

BATCH_SIZE = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', default = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p1_data/val')
parser.add_argument('--label_dir', default = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p1_data/id2label.json')
parser.add_argument('--output_dir', default = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/output_p1/pred_inf.csv')
args = parser.parse_args()

model, preprocess = clip.load('ViT-B/32', device=device)

dataset = pb1_dataset(args.img_dir, preprocess)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# with open(f'{args.label_dir}/id2label.json', 'r') as f:
with open(f'{args.label_dir}', 'r') as f:
    id2label = json.load(f)
label_text = list(id2label.values())

text = torch.cat([clip.tokenize(f"Yes! This is definitely a photo of a {label}") for label in label_text]).to(device)

with open(args.output_dir, 'a') as f:
    f.write('filename,label\n')

with torch.no_grad():
    for image, img_name in tqdm(loader):
        image = image.to(device)

        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=1, keepdim=True)
        text_features /= text_features.norm(dim=1, keepdim=True)

        similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
        pred = similarity[0].argmax()

        with open(args.output_dir, 'a') as f:
            f.write(f'{img_name[0]},{pred}\n')

