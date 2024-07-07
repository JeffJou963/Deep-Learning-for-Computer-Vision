import os
import torch
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from pb_dataloader import pb3
from pb3_model import DANNCNN

# https://github.com/fungtion/DANN_py3

randseed = 1
np.random.seed(randseed)
torch.manual_seed(randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(randseed)

def DANN():
    now = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    # parser.add_argument('--target', default = 'usps')
    parser.add_argument('--target', default = 'svhn')

    # parser.add_argument('--data_dir', default = './hw2_data/digits/usps/test')
    parser.add_argument('--data_dir', default = './hw2_data/digits/svhn/test')
    
    parser.add_argument('--save_dir', default = f'./pb3_output/{now:%Y%m%d_%H%M}')
    args = parser.parse_args()

    device = "cuda:0"
    LR = 2e-4
    batch_size = 256
    n_epoch = 100

    # load data
    # train: source's image & labels in , target: image, val: target

    dataset_source = pb3(f"./hw2_data/digits/mnistm", 'train.csv', 'mnsist')
    dataloader_source = DataLoader(dataset_source, batch_size=batch_size, shuffle=True, num_workers=8)
    # only label
    dataset_target = pb3(f"./hw2_data/digits/{args.target}", 'train.csv', f'{args.target}')
    dataloader_target = DataLoader(dataset_target, batch_size=batch_size, shuffle=True, num_workers=8)

    val_dataset = pb3(f"./hw2_data/digits/{args.target}", 'val.csv', f'{args.target}')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True, num_workers=8)

    pb3_model = DANNCNN()
    # pb3_model.load_state_dict(torch.load('./pb3_output/usps_ep19_acc0.7231.pth'))

    optimizer = torch.optim.AdamW(pb3_model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=0)

    loss_class = nn.CrossEntropyLoss()
    loss_domain = nn.CrossEntropyLoss()

    pb3_model.to(device)
    loss_class.to(device)
    loss_domain.to(device)
    
    for p in pb3_model.parameters():
        p.requires_grad = True

    # train
    for epoch in range(n_epoch):

        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        for i in range(len_dataloader):
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # train source data
            data_source = next(data_source_iter)
            _, s_img, s_label = data_source

            pb3_model.zero_grad()
            domain_label = torch.zeros(len(s_label)).long()

            s_img = s_img.to(device)
            s_label = s_label.to(device)
            domain_label = domain_label.to(device)

            class_output, domain_output = pb3_model(input_data=s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # train target data
            data_target = next(data_target_iter)
            _, t_img, _ = data_target
            domain_label = torch.ones(len(t_img)).long()
            t_img = t_img.to(device)
            domain_label = domain_label.to(device)

            _, domain_output = pb3_model(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            error = err_t_domain + err_s_domain + err_s_label
            error.backward()
            optimizer.step()
        scheduler.step()
        
        print(f'epoch:{epoch} | iter: {i+1}/{len_dataloader} | '
            f'err_s_label: {err_s_label.data.cpu().numpy():.4f} | '
            f'err_s_domain: {err_s_domain.data.cpu().numpy():.4f} | '
            f'err_t_domain: {err_t_domain.data.cpu().item():.4f}')

        # Val
        # with torch.no_grad:
        num_correct = 0
        for _, val_img, val_label in tqdm(val_dataloader):
            # test model using target data
            val_img = val_img.to(device)
            val_label = val_label.to(device)

            class_output, _ = pb3_model(input_data=val_img, alpha=0)
            pred = class_output.argmax(dim=1)                
            # pred = class_output.data.max(1, keepdim=True)[1]
            num_correct += pred.eq(val_label.data.view_as(pred)).sum().item()

        acc = num_correct / len(val_dataloader.dataset)
        print('Accuracy:', f'{acc:.4f}')
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(pb3_model.state_dict(), f'{args.save_dir}/{args.target}_ep{epoch}_acc{acc:.4f}.pth')

if __name__ == '__main__':
    DANN()