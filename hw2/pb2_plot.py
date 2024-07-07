
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
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid


from pb1_model import ContextUnet
from pb2_model import DDIM
from UNet import UNet
from pb_dataloader import FACE

randseed = 1
np.random.seed(randseed)
torch.manual_seed(randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(randseed)



def slerp(z1, z2, alpha):
    theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
    return (
        torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
        + torch.sin(alpha * theta) / torch.sin(theta) * z2
    )


def sample_face():
    now = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default = 'alph')
    parser.add_argument('--prenoise', default = './hw2_data/face/noise')
    parser.add_argument('--save_dir', default = f'./pb2_output/face_{now:%Y%m%d_%H%M}/')
    parser.add_argument('--pretrained', default = './hw2_data/face/UNet.pt')
    args = parser.parse_args()

    device = "cuda"
    n_T = 1000 # time steps

    ddim = DDIM(n_T)
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.pretrained))

    dataset = FACE(f"./hw2_data/face")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 10 face images + loss
    if args.mode == 'ten':
        loss_ema = 0
        for index, gt, noise in tqdm(dataloader):
            index = index.item()        
            x_gen =  ddim.sample(model=model, batch_size=1, channels=3,ddim_timesteps=50, ddim_eta=0.0, clip_denoised=True, noise=noise)
            save_image(x_gen, args.save_dir + f'{index:02}.png', normalize=True, range=(-1, 1))

            x_gen = x_gen.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            criterion = nn.MSELoss()
            loss = criterion(x_gen, gt)
            loss_ema += loss.item()
        print('loss: ', loss_ema)

    # (2-1) different eta
    elif args.mode == 'eta':
        count = 0
        gen_images= []
        etas = [0.0, 0.25, 0.5, 0.75, 1.0]
        for index, gt, noise in tqdm(dataloader):
            count += 1
            for eta in etas:
                x_gen =  ddim.sample(model=model, batch_size=1, channels=3,ddim_timesteps=50, ddim_eta=eta, clip_denoised=True, noise=noise)
                gen_images.append(x_gen)
            
            if count == 4:
                concat_images = torch.cat(gen_images, dim=0)
                save_image(concat_images, 'different_etas.png', nrow=len(etas), normalize=True, range=(-1, 1))
                break

    # (2-2) spherical linear & linear interpolation
    else:
        n=[]
        gen_images= []
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for _, _, noise in iter(dataloader):
            n.append(noise)
            if len(n) ==2:
                break

        for alpha in alphas:
            # slerp
            # noise = slerp(n[0], n[1], alpha)
            # linear 
            noise = (1-alpha)*n[0] + alpha*n[1]

            x_gen =  ddim.sample(model=model, batch_size=1, channels=3,ddim_timesteps=50, ddim_eta=0.0, clip_denoised=True, noise=noise)
            x_gen = x_gen.to(device)
            gen_images.append(x_gen)
        
        concat_images = torch.cat(gen_images, dim=0)
        save_image(concat_images, 'linear_alpha.png', nrow=len(alphas), normalize=True, range=(-1, 1))
        
if __name__ == "__main__":
    sample_face()




