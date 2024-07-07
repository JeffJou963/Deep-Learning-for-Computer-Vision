
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from pb1_model import DDPM, ContextUnet

randseed = 1
np.random.seed(randseed)
torch.manual_seed(randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(randseed)

#ref: https://github.com/TeaPearce/Conditional_Diffusion_MNIST
def sample_mnist():

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='./pb1_output/folder/')
    args = parser.parse_args()

    batch_size = 256
    device = "cuda:0"
    n_classes = 10
    n_feat = 128
    n_T = 1000

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)
    # ddpm.load_state_dict(torch.load("./epoch49_loss0.0160.pth"))
    ddpm.load_state_dict(torch.load("./hw2_1_ckpt.pth"))


    ddpm.eval()
    with torch.no_grad():
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        # all digits x100
        for idx in range(10):
            x_gen, x_gen_store = ddpm.sample_each(100, (3, 28, 28), device, idx, guide_w=2.0)
            for i in range(len(x_gen)):
                save_image(x_gen[i], f'{args.save_dir}/{idx}_{i+1:03d}.png')

if __name__ == "__main__":
    sample_mnist()

