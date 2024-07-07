import os
import sys
import time
import torch
import datetime
import argparse
import numpy as np
import torch.nn as nn
import multiprocessing
from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from byol_pytorch import BYOL

from dataloader import Mini_ImageNet
from tool import set_seed, save_model, load_parameters
from model import Resnet

##### CONSTANTS #####
EPOCH_NUM = 300
BATCH_SIZE = 128
LR = 1e-4
MILESTONE = [30, 50, 80]
# MILESTONE = [5, 10, 20]
NUM_WORKERS = multiprocessing.cpu_count()

def sample_unlabelled_images():
    return torch.randn(20, 3, 128, 128)

def train_classification(model, train_loader, logfile_path, save_dir, learner, optimizer, scheduler, device, current_time):

    model = model.to(device)
    total_train_loss = np.zeros(EPOCH_NUM, dtype=np.float32)
    best_lost = 0.0024

    for epoch in range(EPOCH_NUM):
        # Training
        model.train()
        train_loss = 0.0
        print('=====================')
        print('epoch = {}'.format(epoch))

        for batch, images in enumerate(tqdm(train_loader)):
            # size of data (Batch_size, Channel, H, W)
            images = images.to(device)
            learner = learner.to(device)
            
            loss = learner(images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()

        scheduler.step()

        train_loss = train_loss / len(train_loader.dataset)
        total_train_loss[epoch] = train_loss

        print(f'train loss : {train_loss:.4f} ' )
        print('=====================\n')

        with open(logfile_path, 'a') as f :
            f.write(f'epoch = {epoch}\n', )
            f.write(f'training loss : {train_loss} \n' )
            f.write('===================================\n')

        if epoch == 1:
            save_model(model, optimizer, scheduler, os.path.join(save_dir, current_time, f'epoch_{epoch}.pt')) 
        if epoch%10 == 0:
            save_model(model, optimizer, scheduler, os.path.join(save_dir, current_time, f'epoch_{epoch}.pt')) 
        if epoch == EPOCH_NUM-1:
            save_model(model, optimizer, scheduler, os.path.join(save_dir, current_time, f'epoch_{epoch}.pt')) 
        if  train_loss < best_lost:
            best_lost = train_loss
            save_model(model, optimizer, scheduler, os.path.join(save_dir, f'best_model.pt')) 
            save_model(model, optimizer, scheduler, os.path.join(save_dir, current_time, f'best_model.pt')) 
            save_model(model, optimizer, scheduler, os.path.join(save_dir, current_time, f'epoch_{epoch}.pt')) 

def test_classification(model, test_loader, learner, csv_dir, device, pretrained):
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        labels = []

        for batch, (images, label, img_name) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            learner = learner.to(device)

            output = model(images)

            loss = learner(images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1)
            labels.extend(label.detach().cpu().tolist())
            correct_prediction += (pred.eq(label.view_as(pred)).sum().item())

            with open(csv_dir, 'a') as f :
                if batch == 0:
                    f.write('filename,label\n')
            with open(csv_dir, 'a') as f :
                for i in range(len(img_name)):
                    f.write('{},{}\n'.format(img_name[i].split('/')[-1], pred[i]))

    test_acc = correct_prediction / len(test_loader.dataset)
    print('====================')
    print(f' test loss = {test_loss:.4f}' )
    print('====================')

def main():    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_type', default = 'modelB', help = 'modelA or modelB')
    parser.add_argument('--mode', default = 'train', help = 'train or test')
    parser.add_argument('--test_dir', default = './hw1_data/p2_data/office')
    parser.add_argument('--csv_dir', default = './output/pred.csv', help = 'train or test')
    args = parser.parse_args()

    set_seed(9527)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    resnet = models.resnet50(pretrained=False)
    # resnet = Resnet()

    learner = BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool',
        use_momentum = False       # turn off momentum in the target encoder
    )

    if args.mode == 'train':
        checkpoint_path = './pb2_self_supervised/train_checkpoints/best_model.pt'
        load_parameters(resnet, checkpoint_path, device)
        save_dir = './pb2_self_supervised/train_checkpoints/'
        os.makedirs(save_dir, exist_ok=True)
        pretrained = False

        logfile_dir = os.path.join(save_dir, current_time)
        os.makedirs(logfile_dir, exist_ok=True)  # Create the directory for the current timestamp
        logfile_path = os.path.join(logfile_dir, 'logfile.txt')

        train_set = Mini_ImageNet('./hw1_data/p2_data/mini/train')
        train_loader = DataLoader(train_set, BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)

        # optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)
        optimizer = torch.optim.AdamW(learner.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=0)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONE, gamma=0.1)
        
        train_classification(model = resnet,
            train_loader   = train_loader,
            logfile_path   = logfile_path,
            save_dir       = save_dir,
            learner        = learner,
            optimizer      = optimizer,
            scheduler      = scheduler,
            device         = device,
            current_time   = current_time
            )

    else: # test
        
        model = models.resnet50(pretrained=True)
        checkpoint_path = './pb2_self_supervised/train_checkpoints/best_model.pt'
        load_parameters(model, checkpoint_path, device)

        test_set = Mini_ImageNet(args.test_dir)
        test_loader = DataLoader(test_set, BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)

        test_classification(model = model,
            test_loader    = test_loader,
            learner        = learner,
            csv_dir        = args.csv_dir,
            device         = device,
            )
        
if __name__ == '__main__':
    main()