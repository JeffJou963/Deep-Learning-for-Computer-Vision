import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation import deeplabv3_resnet50

from tool import load_parameters

class Residual_Block(nn.Module):
    def __init__(self, in_channels):
        super(Residual_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.activ = nn.ReLU()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.activ(x1)
        x3 = self.conv(x2)
        x4 = x3 + x
        return self.activ(x4)


class modelA(nn.Module):
    def __init__(self):
        super(modelA, self).__init__()
        self.resnet50 = models.resnet50(pretrained = False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 128)
        self.fc2 = nn.Linear(128, 50)
    
    def forward(self, x):
        x = self.resnet50(x)
        Sec_Last_Layer = x
        x = self.fc2(x)

        return x, Sec_Last_Layer


class modelB(nn.Module):
    def __init__(self):
        super(modelB, self).__init__()
        self.resnet = models.resnet152(pretrained = True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 50)
    
    def forward(self, x):
        return self.resnet(x)


class Resnet(nn.Module):
    def __init__(self, setting, device):
        super(Resnet, self).__init__()
        # self.resnet = models.resnet50(pretrained = True)
        self.resnet = models.resnet50(pretrained = False)
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 65)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 65)
        )

        if setting == 'A':
            checkpoint_path = './pb2_self_supervised/checkpoints_A/best_model.pt'
            load_parameters(self.resnet, checkpoint_path, device)
            
        elif setting == 'B':
            checkpoint_path = './hw1_data/p2_data/pretrain_model_SL.pt'
            print(f'Loading model parameters from {checkpoint_path}...')
            state = torch.load(checkpoint_path, map_location=device)
            self.resnet.load_state_dict(state)
            
        elif setting == 'C':    
            checkpoint_path = './pb2_self_supervised/train_checkpoints/best_model.pt'
            load_parameters(self.resnet, checkpoint_path, device)
            
        elif setting == 'D':
            checkpoint_path = './hw1_data/p2_data/pretrain_model_SL.pt'
            print(f'Loading model parameters from {checkpoint_path}...')
            state = torch.load(checkpoint_path, map_location=device)
            self.resnet.load_state_dict(state)
            for param in self.resnet.parameters():
                param.requires_grad = False   

        elif setting == 'E':    
            checkpoint_path = './pb2_self_supervised/train_checkpoints/best_model.pt'
            # checkpoint_path = './pb2_self_supervised/checkpoints_E/best_model.pt'
            load_parameters(self.resnet, checkpoint_path, device)
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            pass

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return x

# ref: https://github.com/nikhitmago/semantic-segmentation-transfer-learning/blob/master/fcn.ipynb
class FCN32s(nn.Module):
    def __init__(self, num_classes):
        super(FCN32s, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        # for param in vgg16.features.parameters():
        #     param.requires_grad = False

        self.features = vgg16.features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(4096, num_classes, 1),
            nn.ConvTranspose2d(num_classes, num_classes, 224, stride=32)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Deeplabv3(nn.Module):
    def __init__(self, num_classes):
        super(Deeplabv3, self).__init__()
        self.deeplabv3 = deeplabv3_resnet50(pretrained=False)
        self.deeplabv3.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    def forward(self, x):
        return self.deeplabv3(x)