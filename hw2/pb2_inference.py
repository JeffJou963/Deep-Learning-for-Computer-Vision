
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid

from pb2_model import DDIM
from UNet import UNet
from pb_dataloader import FACE, FACE_inf

randseed = 1
np.random.seed(randseed)
torch.manual_seed(randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(randseed)

def sample_face():

    parser = argparse.ArgumentParser()
    parser.add_argument('--prenoise', default = './hw2_data/face/noise')
    parser.add_argument('--save_dir', default = f'./pb2_output/')
    parser.add_argument('--pretrained', default = './hw2_data/face/UNet.pt')
    args = parser.parse_args()

    device = "cuda"
    n_T = 1000

    ddim = DDIM(n_T)
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.pretrained))

    dataset = FACE_inf(args.prenoise)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # inf: generate 10 images
    for index, noise in tqdm(dataloader):
        index = index.item()        
        x_gen =  ddim.sample(model=model, batch_size=1, channels=3,ddim_timesteps=50, ddim_eta=0.0, clip_denoised=True, noise=noise)
        save_image(x_gen, f'{args.save_dir}/{index:02}.png', normalize=True, range=(-1, 1))


if __name__ == "__main__":
    sample_face()




