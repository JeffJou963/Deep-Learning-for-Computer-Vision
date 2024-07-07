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

from model import modelA, modelB
from dataloader import pb1_dataset
from tool import set_seed, save_model, load_parameters, visualize_pca, visualize_tsne

##### CONSTANTS #####
EPOCH_NUM = 50
BATCH_SIZE = 32
LR = 0.01
MILESTONE = [12, 24, 32]
# MILESTONE = [5, 10, 20]
NUM_WORKERS = multiprocessing.cpu_count()

def train_classification(model, train_loader, val_loader, logfile_path, save_dir, criterion, optimizer, scheduler, device, pretrained, current_time):

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
        print('epoch = {}'.format(epoch)) 

        for batch, (image, label, img_name) in enumerate(tqdm(train_loader)):
            # size of data (Batch_size, Channel, H, W)
            image = image.to(device)
            label = label.to(device)

            # input -(model)-> output, +label -> loss
            if pretrained: # modelB
                output = model(image)
            else:
                output, SecondLastFeture = model(image)

            optimizer.zero_grad()
            loss = criterion(output, label)            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # argmax(dim=1) is the class with highest predicted score
            pred = output.argmax(dim=1)
            # pred.eq(label.view_as(pred)): boolean, .item(): convert to integer
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
                
                if pretrained: # modelB
                    output = model(image)
                else:
                    output, SecondLastFeture = model(image)
                loss = criterion(output, label)
                val_loss += loss.item()

                pred = output.argmax(dim=1)
                correct_prediction += (pred.eq(label.view_as(pred)).sum().item())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_prediction / len(val_loader.dataset)
        total_val_loss[epoch] = val_loss
        total_val_acc[epoch] = val_acc

        print('*'*10)
        print(f'train loss : {train_loss:.4f} ', f' train acc = {train_acc:.4f}' )
        print(f'val loss : {val_loss:.4f} ', f' val acc = {val_acc:.4f}' )
        print('========================\n')

        with open(logfile_path, 'a') as f :
            f.write(f'epoch = {epoch}\n', )
            f.write(f'training loss : {train_loss}  train acc = {train_acc}\n' )
            f.write(f'val loss : {val_loss}  val acc = {val_acc}\n' )
            f.write('============================\n')

        # save the last epoch and best model
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


    x = range(0, EPOCH_NUM)
    total_train_acc = total_train_acc.tolist()
    total_train_loss = total_train_loss.tolist()
    total_val_acc = total_val_acc.tolist()
    total_val_loss = total_val_loss.tolist()

    # Plot Learning Curve
    # Consider the function plot_learning_curve(x, y) above

    #acc
    plt.figure()
    plt.title('accuracy',fontsize=10)
    plt.xlabel('epoch', fontsize=10)
    plt.ylabel('acc', fontsize=10)
    train_acc_curve = plt.plot(x, total_train_acc, 'b')
    val_acc_curve = plt.plot(x, total_val_acc, 'g')
    plt.legend(["training", "validation"], loc='best')
    plt.savefig(os.path.join(save_dir, 'acc_{}.png'.format(best_acc)))
    plt.close()

    #loss
    plt.figure()
    plt.title('loss',fontsize=10)
    plt.xlabel('epoch', fontsize=10)
    plt.ylabel('loss', fontsize=10)
    train_loss_curve = plt.plot(x, total_train_loss, 'b')
    val_loss_curve = plt.plot(x, total_val_loss, 'g')
    plt.legend(["training", "validation"], loc='best')
    plt.savefig(os.path.join(save_dir, 'loss_{}.png'.format(best_acc)))
    plt.close()


def test_classification(model, test_loader, csv_dir, device, pretrained):
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        correct_prediction = 0
        Sec_Last_Layers = torch.zeros((0, 128), dtype = torch.float32)
        # Sec_Last_Layers = torch.zeros((0, 256), dtype = torch.float32)
        labels = []

        for batch, (image, label, img_name) in enumerate(tqdm(test_loader)):
            image = image.to(device)
            label = label.to(device)

            if pretrained: # modelB
                output = model(image)
            else:
                output, Sec_Last_Layer = model(image)            
                Sec_Last_Layers = torch.cat((Sec_Last_Layers, Sec_Last_Layer.detach().cpu()), 0)

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
    # print('====================')
    # print(f' test acc = {test_acc:.4f}' )
    # print('====================')

    # visualize_pca(Sec_Last_Layers.numpy(), labels, num_components=2)
    # visualize_tsne(Sec_Last_Layers.numpy(), labels, num_components=2)

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default = 'modelB', help = 'modelA or modelB')
    parser.add_argument('--mode', default = 'test', help = 'train or test')
    parser.add_argument('--test_dir', default = './hw1_data/p1_data/val_50')
    parser.add_argument('--csv_dir', default = './output/pred_p1.csv', help = 'train or test')
    args = parser.parse_args()

    set_seed(9527)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    # TRAIN & VAL OR TEST 
    # MODEL / DATALOADER / LOSS
    if args.mode == 'train':
        if args.model_type == 'modelA':
            model = modelA()
            # checkpoint_path = './pb1_classification/modelA_checkpoints/best_model.pt'
            # load_parameters(model, checkpoint_path, device)
            save_dir = './pb1_classification/modelA_checkpoints/'
            os.makedirs(save_dir, exist_ok=True)
            pretrained = False

        else :
            model = modelB()
            # checkpoint_path = './pb1_classification/modelB_checkpoints/best_model.pt'
            checkpoint_path = './pb1_classification/modelB_checkpoints/epoch_99.pt'
            load_parameters(model, checkpoint_path, device)
            save_dir = './pb1_classification/modelB_checkpoints/'
            os.makedirs(save_dir, exist_ok=True)
            pretrained = True

        logfile_dir = os.path.join(save_dir, current_time)
        os.makedirs(logfile_dir, exist_ok=True)  # Create the directory for the current timestamp
        logfile_path = os.path.join(logfile_dir, 'logfile.txt')

        train_set = pb1_dataset('./hw1_data/p1_data/train_50', 'train')
        val_set = pb1_dataset('./hw1_data/p1_data/val_50', 'train')
        train_loader = DataLoader(train_set, BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
        val_loader = DataLoader(val_set, BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)

        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
   
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-6, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONE, gamma=0.1)
        
        train_classification(model = model,
            train_loader   = train_loader,
            val_loader     = val_loader,
            logfile_path    = logfile_path,
            save_dir       = save_dir,
            criterion      = criterion,
            optimizer      = optimizer,
            scheduler      = scheduler,
            device         = device,
            pretrained     = pretrained,
            current_time   = current_time
            )

    else: # test
        if args.model_type == 'modelA':
            model = modelA()
            pretrained = False
        else:
            model = modelB()
            pretrained = True
        
        checkpoint_path = './hw1_1.pt'
        # checkpoint_path = f'./pb1_classification/{args.model_type}_checkpoints/epoch_99.pt'
        load_parameters(model, checkpoint_path, device)

        test_set = pb1_dataset(args.test_dir, 'test')
        test_loader = DataLoader(test_set, BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)

        test_classification(model = model,
            test_loader    = test_loader,
            csv_dir        = args.csv_dir,
            device         = device,
            pretrained     = pretrained
            )
        
if __name__ == '__main__':
    main()