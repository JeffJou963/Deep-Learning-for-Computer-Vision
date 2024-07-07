import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Tuple
from torch.utils.data import DataLoader

from pb_dataloader import pb3_inf
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
    device = "cuda:0"
    batch_size = 256

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_image', default = './hw2_data/digits/usps/data')
    parser.add_argument('--save_dir', default = './usps_test.csv')
    args = parser.parse_args()

    pb3_model = DANNCNN()
    if 'svhn' in args.test_image:
        dataset = pb3_inf(args.test_image, 'svhn')
        pb3_model.load_state_dict(torch.load('./svhn_ep88_acc0.4344.pth'))
    elif 'usps' in args.test_image:
        dataset = pb3_inf(args.test_image, 'usps')
        pb3_model.load_state_dict(torch.load('./usps_ep93_acc0.8575.pth'))
    else:
        print('error??')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = False, num_workers=8)
    pb3_model.to(device)    

    with open(args.save_dir, 'a') as f :
        f.write('image_name,label\n')

    for img_name, img, label in tqdm(dataloader):
        img = img.to(device)
        label = label.to(device)

        class_output, _ = pb3_model(input_data=img, alpha=0)
        pred = class_output.argmax(dim=1)

        with open(args.save_dir, 'a') as f :
            for i in range(len(img_name)):
                f.write('{},{}\n'.format(img_name[i].split('/')[-1], pred[i]))

if __name__ == '__main__':
    DANN()