import os
import json
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from tokenizer import BPETokenizer

class pb1_dataset(Dataset):
    def __init__ (self, data_dir, preprocess):
        self.images = []
        self.preprocess = preprocess

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                self.images.append(os.path.join(root, file))

    def __getitem__(self, index):
        image = self.preprocess(Image.open(self.images[index]))
        img_class = int(self.images[index].split('/')[-1].split('_')[0])
        img_name = self.images[index].split('/')[-1]
        return image, img_class, img_name

    def __len__(self):
        return len(self.images)
    

class pb2_dataset(Dataset):
    def __init__(self, data_dir, json_dir, mode):
        self.mode = mode
        self.data_dir = data_dir
        self.json_dir = json_dir

        self.captions = []
        self.image_dir = []
        self.tokenizer = BPETokenizer('./encoder.json', './vocab.bpe')
        self.max_cap_length = 60

        with open(json_dir) as f:
            json_file = json.load(f)
        annotations = json_file['annotations']
        images = json_file['images']

        caption_index = 0
        for annotation in annotations:
            if mode == 'val':
                if caption_index % 5 == 0:
                    self.captions.append(annotation['caption'])
                    for image in images:
                        if image['id'] == annotation['image_id']:
                            self.image_dir.append(os.path.join(data_dir, image['file_name']))
                caption_index += 1

            else:
                self.captions.append(annotation['caption'])
                for image in images:
                    if image['id'] == annotation['image_id']:
                        self.image_dir.append(os.path.join(data_dir, image['file_name']))
    
        self.train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    

    def __getitem__(self, index):
        image_id = self.image_dir[index]
        image = Image.open(image_id).convert('RGB')
        
        if self.mode == 'train':
            image = self.train_transform(image)
            caption = self.captions[index]
            caption_in = self.tokenizer.encode(caption)
            caption_in.insert(0, 50256)
            caption_in = np.array(caption_in)
            caption_in = np.pad(caption_in, (0,self.max_cap_length - len(caption_in)), mode='constant', constant_values=50256)
            caption_in = torch.LongTensor(caption_in)

            caption_gt = self.tokenizer.encode(caption)
            caption_gt.append(50256)
            caption_gt = np.array(caption_gt)
            caption_gt = np.pad(caption_gt, (0,self.max_cap_length - len(caption_gt)), mode='constant', constant_values=-100)
            caption_gt = torch.LongTensor(caption_gt)

            img_file_name = image_id.split('/')[-1].split('.')[0]
            return img_file_name, image, caption_in, caption_gt 

        elif self.mode == 'val':
            image = self.test_transform(image)
            caption = self.captions[index]
            caption_in = self.tokenizer.encode(caption)
            caption_in.insert(0, 50256)
            caption_in = np.array(caption_in)
            caption_in = np.pad(caption_in, (0,self.max_cap_length - len(caption_in)), mode='constant', constant_values=50256)
            caption_in = torch.LongTensor(caption_in)

            caption_gt = self.tokenizer.encode(caption)
            caption_gt.append(50256)
            caption_gt = np.array(caption_gt)
            caption_gt = np.pad(caption_gt, (0,self.max_cap_length - len(caption_gt)), mode='constant', constant_values=-100)
            caption_gt = torch.LongTensor(caption_gt)

            img_file_name = image_id.split('/')[-1].split('.')[0]
            return img_file_name, image, caption_in, caption_gt 

        else:
            image = self.test_transform(image)
            img_file_name = image_id.split('/')[-1].split('.')[0]
            # img_file_name = image_id.split('/')[-1]
            return img_file_name, image
    
    def __len__(self):
        return len(self.captions)
    
if __name__ == '__main__':
    BATCHSIZE = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test')
    # parser.add_argument('--data_dir', default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/images/train')
    parser.add_argument('--data_dir', default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/images/val')
    parser.add_argument('--out_dir', default='./output_p2/pred.json')
    parser.add_argument('--decoder_weights', default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data')
    args = parser.parse_args()

    train_json_dir = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/train.json'
    val_json_dir = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/val.json'

    # train_dataset = pb2_dataset(args.data_dir, train_json_dir,'train')
    # train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
    val_dataset = pb2_dataset(args.data_dir, val_json_dir, 'val')
    val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # iter = iter(train_loader)
    iter = iter(val_loader)
    for i in range(5):
        (ifn,), image, caption_in, caption_gt = next(iter)

        # img_file_name = img_file_name.to(device)
        print(ifn)
        # print(img_file_name.shape)

        # print(caption_in)
        # print(image.shape)  # (b,3,224,224)
        # print(caption_in.shape) # (b, max_length)
        # print(caption_in.shape) # (b, max_length)
