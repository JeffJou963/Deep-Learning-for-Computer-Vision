import os
import sys
import time
import torch
import datetime
import argparse
import numpy as np
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from mean_iou_evaluate import mean_iou_score
from model import FCN32s, Deeplabv3
from dataloader import pb3_dataset
from tool import set_seed, save_model, load_parameters

##### CONSTANTS #####
EPOCH_NUM = 40
BATCH_SIZE = 8
# LR = 0.01
LR = 0.001
MILESTONE = [8, 13, 20]
NUM_WORKERS = 0

def train_segmentation(model, train_loader, val_loader, logfile_path, save_dir, criterion, optimizer, scheduler, device, current_time):

    model = model.to(device)
    total_train_loss = np.zeros(EPOCH_NUM, dtype=np.float32)
    total_train_iou  = np.zeros(EPOCH_NUM, dtype=np.float32)
    total_val_loss   = np.zeros(EPOCH_NUM, dtype=np.float32)
    total_val_iou    = np.zeros(EPOCH_NUM, dtype=np.float32)
    best_iou = 0

    for epoch in range(EPOCH_NUM):
        # Training
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        preds = []
        gts  = []
 
        print('epoch = {}'.format(epoch)) 

        for batch, (image, mask,_) in enumerate(tqdm(train_loader)):
            # size of data (Batch_size, Channel, H, W)
            image = image.to(device)
            mask = mask.to(device)

            # if deeplav3 == True:
            output = model(image)
            loss = criterion(output['out'], mask)
            # loss = criterion(output, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # pred = output.argmax(dim=1).cpu().numpy()
            pred = output['out'].argmax(dim=1).cpu().numpy()
            gt = mask.cpu().numpy()
            for p, g in zip(pred, gt):
                preds.append(p)
                gts.append(g)

        preds = np.asarray(preds)
        gts   = np.asarray(gts)
        scheduler.step()
        
        train_iou = mean_iou_score(preds, gts)
        train_loss = train_loss / len(train_loader.dataset)
        total_train_iou[epoch] = train_iou
        total_train_loss[epoch] = train_loss
     
        # Validation
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_iou = 0.0
            preds = []
            gts  = []
 
            for batch, (image, mask, _) in enumerate(tqdm(val_loader)):
                image = image.to(device)
                mask = mask.to(device)
                
                output = model(image)
    
                # loss = criterion(output, mask)
                # pred = output.argmax(dim=1).cpu().numpy()
                loss = criterion(output['out'], mask)
                pred = output['out'].argmax(dim=1).cpu().numpy()

                val_loss += loss.item()
                gt = mask.cpu().numpy()
                for p, g in zip(pred, gt):
                    preds.append(p)
                    gts.append(g)
            preds = np.asarray(preds)
            gts = np.asarray(gts)
            val_iou = mean_iou_score(preds, gts)

        val_loss = val_loss / len(val_loader.dataset)
        total_val_loss[epoch] = val_loss
        total_val_iou[epoch] = val_iou

        print('*'*10)
        print(f'train loss : {train_loss:.4f} ', f' train iou = {train_iou:.4f}' )
        print(f'val loss : {val_loss:.4f} ', f' val iou = {val_iou:.4f}' )
        print('========================\n')

        with open(logfile_path, 'a') as f :
            f.write(f'epoch = {epoch}\n', )
            f.write(f'training loss : {train_loss}  train iou = {train_iou}\n' )
            f.write(f'val loss : {val_loss}  val iou = {val_iou}\n' )
            f.write('============================\n')

        # save the last epoch and best model
        if epoch == 1:
            save_model(model, optimizer, scheduler, os.path.join(save_dir, current_time, f'epoch_{epoch}.pt')) 
        if epoch%10 == 0:
            save_model(model, optimizer, scheduler, os.path.join(save_dir, current_time, f'epoch_{epoch}.pt')) 
        if epoch == EPOCH_NUM-1:
            save_model(model, optimizer, scheduler, os.path.join(save_dir, current_time, f'epoch_{epoch}.pt')) 
        if  val_iou > best_iou:
            best_iou = val_iou
            save_model(model, optimizer, scheduler, os.path.join(save_dir,current_time, f'epoch_{epoch}.pt')) 
            save_model(model, optimizer, scheduler, os.path.join(save_dir,current_time, f'best_model.pt')) 

def test_segmentation(model, test_loader, csv_dir, device):
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        correct_prediction = 0
        preds = []
        gts  = []

        for batch, (image, mask, image_name) in enumerate(tqdm(test_loader)):
            image = image.to(device)
            mask = mask.to(device)
            
            output = model(image)
            pred = output['out'].argmax(dim=1).cpu().numpy()
            gt = mask.cpu().numpy()

            for p, g in zip(pred, gt):
                preds.append(p)
                gts.append(g)

            # print('image_shape: ', image.shape)
            # print('mask_shape: ', mask.shape)
            # print('pred_shape: ', pred.shape)

            # len(pred) = batch size
            for i in range(len(pred)):
                color_mapping = {
                # class_label: color
                    0: (255, 0, 0),     # Red for Class 0
                    1: (0, 255, 0),     # Green for Class 1
                    2: (0, 0, 255),     # Blue for Class 2
                    3: (255, 255, 0),   # Yellow for Class 3
                    4: (255, 0, 255),   # Magenta for Class 4
                    5: (0, 255, 255),   # Cyan for Class 5
                    6: (128, 128, 128)  # Gray for Class 6 (or any other distinct color)
                }
                pred_rgb = np.zeros((pred.shape[1], pred.shape[2], 3), dtype=np.uint8)
                for class_label, color in color_mapping.items():
                    pred_rgb[pred[i,:,:] == class_label] = color

                pred_rgb_pil = Image.fromarray(pred_rgb)
                os.makedirs(csv_dir, exist_ok=True)
                image_index = image_name[i]
                # image_index: x./hw1_data/.../xxxx_sat.jpg
                image_index = os.path.splitext(os.path.basename(image_index))[0].split('_')[0]

                # if (image_index == '0013' or image_index == '0062' or image_index =='0104'):
                #     # out_path = os.path.join(csv_dir, f"{image_index}_early.png")
                #     # out_path = os.path.join(csv_dir, f"{image_index}_middle.png")
                #     # out_path = os.path.join(csv_dir, f"{image_index}_final.png")
                #     pred_rgb_pil.save(out_path)

                out_path = os.path.join(csv_dir, f"{image_index}_mask.png")
                pred_rgb_pil.save(out_path)

        preds = np.asarray(preds)
        gts = np.asarray(gts)
        val_iou = mean_iou_score(preds, gts)

    print('========================')
    print(f' test iou = {val_iou:.4f}' )
    print('========================\n')



def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default = 'B', help = 'modelA or modelB')
    parser.add_argument('--mode', default = 'test', help = 'train or test')
    parser.add_argument('--test_dir', default = './hw1_data/p3_data/validation')
    parser.add_argument('--csv_dir', default = './output_p3', help = 'train or test')
    args = parser.parse_args()

    set_seed(9527)
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    # TRAIN & VAL OR TEST 
    # MODEL / DATALOADER / LOSS
    if args.mode == 'train':
        if args.model_type == 'modelA':
            model = FCN32s(7)
            save_dir = './pb3_segmentation/modelA_checkpoints/'
            os.makedirs(save_dir, exist_ok=True)

        else :
            model = Deeplabv3(7)
            checkpoint_path = './pb3_segmentation/modelB_checkpoints/epoch_11.pt'
            
            load_parameters(model, checkpoint_path, device)
            save_dir = './pb3_segmentation/modelB_checkpoints/'
            os.makedirs(save_dir, exist_ok=True)

        logfile_dir = os.path.join(save_dir, current_time)
        os.makedirs(logfile_dir, exist_ok=True)  # Create the directory for the current timestamp
        logfile_path = os.path.join(logfile_dir, 'logfile.txt')

        train_set = pb3_dataset('./hw1_data/p3_data/train', aug=True)
        val_set = pb3_dataset('./hw1_data/p3_data/validation', aug=True)
        train_loader = DataLoader(train_set, BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
        val_loader = DataLoader(val_set, BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4, nesterov=True)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONE, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=0)
        
        train_segmentation(model = model,
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
        model = Deeplabv3(7)
        checkpoint_path = './hw1_3.pt'
        # checkpoint_path = './pb3_segmentation/modelB_checkpoints/epoch_11.pt'
        load_parameters(model, checkpoint_path, device)

        test_set = pb3_dataset(args.test_dir, aug=False)
        test_loader = DataLoader(test_set, BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)
        
        test_segmentation(model = model,
            test_loader    = test_loader,
            csv_dir        = args.csv_dir,
            device         = device
        )
        
if __name__ == '__main__':
    main()