import os
import sys
import time
import torch
import datetime
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
# from byol_pytorch import BYOL

from dataloader import Office
from tool import set_seed, save_model, load_parameters
from model import Resnet

##### CONSTANTS #####
EPOCH_NUM = 100
BATCH_SIZE = 64
LR = 1e-3
MILESTONE = [30, 50, 80]
# MILESTONE = [5, 10, 20]
# NUM_WORKERS = multiprocessing.cpu_count()
NUM_WORKERS = 0

def sample_unlabelled_images():
    return torch.randn(20, 3, 128, 128)

def train_classification(model, train_loader,val_loader, logfile_path, save_dir, criterion, optimizer, scheduler, device, current_time):

    model = model.to(device)
    total_train_loss = np.zeros(EPOCH_NUM, dtype=np.float32)
    total_train_acc = np.zeros(EPOCH_NUM, dtype=np.float32)
    total_val_loss = np.zeros(EPOCH_NUM, dtype=np.float32)
    total_val_acc = np.zeros(EPOCH_NUM, dtype=np.float32)
    best_acc = 0

    for epoch in range(EPOCH_NUM):
        # Training
        model.train()
        train_loss = 0.0
        correct_prediction = 0 
        print('=====================')
        print('epoch = {}'.format(epoch))

        for batch, (image, label, img_name) in enumerate(tqdm(train_loader)):
            # size of data (Batch_size, Channel, H, W)
            image = image.to(device)
            label = label.to(device)
            output = model(image)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            pred = output.argmax(dim=1)

            correct_prediction += (pred.eq(label.view_as(pred)).sum().item())

        scheduler.step()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct_prediction / len(train_loader.dataset)
        total_train_loss[epoch] = train_loss
        total_train_acc[epoch] = train_acc

        # Validation
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            correct_prediction = 0 

            for batch, (image, label, img_name) in enumerate(tqdm(val_loader)):
                image = image.to(device)
                label = label.to(device)
                
                output = model(image)
                loss = criterion(output, label)
                val_loss += loss.item()

                pred = output.argmax(dim=1)
                correct_prediction += (pred.eq(label.view_as(pred)).sum().item())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_prediction / len(val_loader.dataset)
        total_val_loss[epoch] = val_loss
        total_val_acc[epoch] = val_acc

        print(f'train loss : {train_loss:.4f} ', f' train acc = {train_acc:.4f}' )
        print(f'val loss : {val_loss:.4f} ', f' val acc = {val_acc:.4f}' )
        print('=====================\n')

        with open(logfile_path, 'a') as f :
            f.write(f'epoch = {epoch}\n', )
            f.write(f'training loss : {train_loss}  train acc = {train_acc}\n' )
            f.write(f'val loss : {val_loss}  val acc = {val_acc}\n' )
            f.write('===================================\n')

        if epoch == 1:
            save_model(model, optimizer, scheduler, os.path.join(save_dir, current_time, f'epoch_{epoch}.pt')) 
        if epoch%10 == 0:
            save_model(model, optimizer, scheduler, os.path.join(save_dir, current_time, f'epoch_{epoch}.pt')) 
        if epoch == EPOCH_NUM-1:
            save_model(model, optimizer, scheduler, os.path.join(save_dir, current_time, f'epoch_{epoch}.pt')) 
        if  val_acc > best_acc:
            best_acc = val_acc
            save_model(model, optimizer, scheduler, os.path.join(save_dir, current_time, f'best_model.pt')) 
            save_model(model, optimizer, scheduler, os.path.join(save_dir, current_time, f'epoch_{epoch}.pt')) 


def test_classification(model, test_loader, csv_dir, device):
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        correct_prediction = 0
        labels = []

        for batch, (image, label, img_name) in enumerate(tqdm(test_loader)):
            image = image.to(device)
            label = label.to(device)

            output = model(image)

            pred = output.argmax(dim=1)
            labels.extend(label.detach().cpu().tolist())
            correct_prediction += (pred.eq(label.view_as(pred)).sum().item())

            with open(csv_dir, 'a') as f :
                if batch == 0:
                    f.write('id,filename,label\n')
            with open(csv_dir, 'a') as f :
                for i in range(len(img_name)):
                    f.write('{},{},{}\n'.format(i, img_name[i].split('/')[-1], pred[i]))

    test_acc = correct_prediction / len(test_loader.dataset)
    print('====================')
    print(f' test acc = {test_acc:.4f}' )
    print('====================')


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', default = 'test', help = 'A-E or test')
    parser.add_argument('--mode', default = 'test', help = 'train or test')
    parser.add_argument('--input_csv', default = './hw1_data/p2_data/office/val.csv')
    parser.add_argument('--test_dir', default = './hw1_data/p2_data/office/val')
    parser.add_argument('--csv_dir', default = './pred_p2_new.csv', help = 'train or test')
    args = parser.parse_args()

    set_seed(9527)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    if args.mode == 'train':
        model = Resnet(args.setting, device)

        save_dir = f'./pb2_self_supervised/checkpoints_{args.setting}/'
        os.makedirs(save_dir, exist_ok=True)

        logfile_dir = os.path.join(save_dir, current_time)
        os.makedirs(logfile_dir, exist_ok=True)  # Create the directory for the current timestamp
        logfile_path = os.path.join(logfile_dir, 'logfile.txt')

        train_set = Office('./hw1_data/p2_data/office/train', 'train')
        val_set = Office('./hw1_data/p2_data/office/val', 'train')
        train_loader = DataLoader(train_set, BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
        val_loader = DataLoader(val_set, BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)

        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=0)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONE, gamma=0.1)
        
        train_classification(model = model,
            train_loader   = train_loader,
            val_loader     = val_loader,
            logfile_path    = logfile_path,
            save_dir       = save_dir,
            criterion      = criterion,
            optimizer      = optimizer,
            scheduler      = scheduler,
            device         = device,
            current_time   = current_time
            )

    else: # test
        model = Resnet(args.setting, device)
        checkpoint_path = './hw1_2.pt'
        # checkpoint_path = './pb2_self_supervised/checkpoints_C/epoch_4.pt'
        # checkpoint_path = './pb2_self_supervised/train_checkpoints/best_model.pt'
        load_parameters(model, checkpoint_path, device)

        test_set = Office(args.test_dir, 'test')
        test_loader = DataLoader(test_set, BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)

        test_classification(model = model,
            test_loader    = test_loader,
            csv_dir        = args.csv_dir,
            device         = device,
            )
        
if __name__ == '__main__':
    main()