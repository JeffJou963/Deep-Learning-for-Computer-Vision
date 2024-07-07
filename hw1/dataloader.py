import os 
from PIL import Image 
from torch.utils.data import Dataset
import torchvision.transforms as transforms 
import torch
import numpy as np
import imageio

class pb1_dataset(Dataset):
    def __init__ (self, data_dir, mode='train'):
        self.mode = mode
        self.images = []
        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std  = (0.229, 0.224, 0.225)
            )
        ])
        self.test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std  = (0.229, 0.224, 0.225)
            )
        ])
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.png'):
                    self.images.append(os.path.join(root, file))
    
    def __getitem__(self, index):
        if self.mode == 'train':
            image = self.train_transforms(Image.open(self.images[index]).convert('RGB'))
        else:
            image = self.test_transforms(Image.open(self.images[index]).convert('RGB'))
        label = int(self.images[index].split('/')[-1].split('.')[0].split('_')[0])
        img_name = self.images[index]
        return (image, label, img_name)
    
    def __len__(self):
        return len(self.images)


class pb3_dataset(Dataset):
    def __init__(self, filepath, aug=False):
        self.imgs = []
        self.masks = []
        self.aug = aug
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225))])

        # read the images and masks
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if file.endswith('jpg'):
                    self.imgs.append(os.path.join(root, file))
                if file.endswith('.png'):
                    self.masks.append(os.path.join(root, file))

        self.imgs.sort(), self.masks.sort()

    def __getitem__(self, index):
        img_name = self.imgs[index]

        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.train_transforms(img)

        mask = imageio.imread(self.masks[index])
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        new_mask =  mask.copy()
        new_mask[mask == 3] = 0  # (Cyan: 011) Urban land 
        new_mask[mask == 6] = 1  # (Yellow: 110) Agriculture land 
        new_mask[mask == 5] = 2  # (Purple: 101) Rangeland 
        new_mask[mask == 2] = 3  # (Green: 010) Forest land 
        new_mask[mask == 1] = 4  # (Blue: 001) Water 
        new_mask[mask == 7] = 5  # (White: 111) Barren land 
        new_mask[mask == 0] = 6  # (Black: 000) Unknown 
        new_mask[mask == 4] = 6
        new_mask = torch.from_numpy(new_mask)

        if self.aug:
            img, new_mask = self.augment(img, new_mask)

        return (img, new_mask, img_name)

    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        p_rotate = np.random.uniform()
        p_flip = np.random.uniform()
        
        if p_flip >= 0.5: 
            img, mask = torch.stack((img[0].T, img[1].T, img[2].T)) , mask.T 
        
        if 0 <= p_rotate < 0.25: 
            img, mask = torch.rot90(img, 1, dims = (1,2)), torch.rot90(mask, 1)
        elif 0.25 <= p_rotate < 0.5: 
            img, mask = torch.rot90(img, -1, dims = (1,2)), torch.rot90(mask, -1)
        elif 0.5 <= p_rotate < 0.75: 
            img, mask = torch.rot90(img, 2, dims = (1,2)), torch.rot90(mask, 2)
    
        return img, mask

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class Mini_ImageNet(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.images = [os.path.join(data_dir, f"{i}.jpg") for i in range(len(os.listdir(data_dir)))]

        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale),

            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

class Office(Dataset):
    def __init__ (self, data_dir, mode='test'):
        self.mode = mode
        self.images = []
        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std  = (0.229, 0.224, 0.225)
            )
        ])
        self.test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std  = (0.229, 0.224, 0.225)
            )
        ])
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.images.append(os.path.join(root, file))
    
    def __getitem__(self, index):
        if self.mode == 'train':
            image = self.train_transforms(Image.open(self.images[index]).convert('RGB'))
        else:
            image = self.test_transforms(Image.open(self.images[index]).convert('RGB'))
        label = int(self.images[index].split('/')[-1].split('.')[0].split('_')[0])
        img_name = self.images[index]
        
        # print(f'Index: {index}, Label: {label}, Image Name: {img_name}')
        return (image, label, img_name)
    
    def __len__(self):
        return len(self.images)