import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MNISTM(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # get img_name and label from train.csv
        self.data_info = pd.read_csv(os.path.join(data_dir,'train.csv'))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # ex: img_name = ./hw2_data/digits/mnistm/data/06936.png
        img_name = os.path.join(self.data_dir, 'data', self.data_info.iloc[index, 0])
        img = Image.open(img_name).convert('RGB')
        img = self.transform(img)
        label = int(self.data_info.iloc[index, 1])

        return img, label
        
    def __len__(self):
        return len(self.data_info)

class FACE(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.GTs = []
        self.noises = []
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        for dir, empty, files in os.walk(os.path.join(data_dir, 'GT')):
            for file in files:
                if file.endswith('.png'):
                    self.GTs.append(os.path.join(dir, file))

        for dir, empty, files in os.walk(os.path.join(data_dir, 'noise')):
            for file in files:
                if file.endswith('.pt'):
                    self.noises.append(os.path.join(dir, file))

        self.GTs.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.noises.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    def __getitem__(self, index):
        GT = self.transforms(Image.open(self.GTs[index]).convert('RGB'))
        noise = torch.load(self.noises[index])
        return index, GT, noise

    def __len__(self):
        return len(self.GTs)


class FACE_inf(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.noises = []
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        for dir, empty, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.pt'):
                    self.noises.append(os.path.join(dir, file))

        self.noises.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    def __getitem__(self, index):
        noise = torch.load(self.noises[index])
        return index, noise

    def __len__(self):
        return len(self.noises)
    

class pb3(Dataset):
    def __init__(self, data_dir, csv, data_type):
        self.data_dir = data_dir
        self.data_type = data_type
        # get img_name and label from csv(train/val/test.csv)
        self.data_info = pd.read_csv(os.path.join(data_dir, csv))

        if self.data_type == 'mnistm':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=([0.4631, 0.4666, 0.4195]), std=([0.1979, 0.1845, 0.2083]))
            ])
        elif self.data_type == 'svhn':
            self.transform = transforms.Compose([        
                transforms.ToTensor(),
                transforms.Normalize(mean=([0.4413, 0.4458, 0.4715]), std=([0.1169, 0.1206, 0.1042]))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=([0.2570, 0.2570, 0.2570]), std=([0.3372, 0.3372, 0.3372]))
            ])

    def __getitem__(self, index):
        # ex: img_name = ./hw2_data/digits/mnistm/data/06936.png
        img_name = os.path.join(self.data_dir, 'data', self.data_info.iloc[index, 0])
        img = Image.open(img_name).convert('RGB')
        img = self.transform(img)
          
        label = int(self.data_info.iloc[index, 1])

        return img_name, img, label
        
    def __len__(self):
        return len(self.data_info)
    
class pb3_inf(Dataset):
    def __init__(self, data_dir, data_type):
        self.data_dir = data_dir
        self.data_type = data_type
        self.images = []

        if self.data_type == 'mnistm':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=([0.4631, 0.4666, 0.4195]), std=([0.1979, 0.1845, 0.2083]))
            ])
        elif self.data_type == 'svhn':
            self.transform = transforms.Compose([        
                transforms.ToTensor(),
                transforms.Normalize(mean=([0.4413, 0.4458, 0.4715]), std=([0.1169, 0.1206, 0.1042]))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=([0.2570, 0.2570, 0.2570]), std=([0.3372, 0.3372, 0.3372]))
            ])

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.png'):
                    self.images.append(os.path.join(root, file))
        self.images.sort()
    
    def __getitem__(self, index):
        img_name = self.images[index]
        img = self.transform(Image.open(self.images[index]).convert('RGB'))
        label = int(self.images[index].split('/')[-1].split('.')[0].split('_')[0])

        return img_name, img, label
        
    def __len__(self):
        return len(self.images)
    