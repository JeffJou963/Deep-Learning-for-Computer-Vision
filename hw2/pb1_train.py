
import os
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from matplotlib.animation import FuncAnimation, PillowWriter

from pb1_model import DDPM, ContextUnet
from pb_dataloader import MNISTM

randseed = 1
np.random.seed(randseed)
torch.manual_seed(randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(randseed)

#ref: https://github.com/TeaPearce/Conditional_Diffusion_MNIST
def train_mnist():

    batch_size = 256
    device = "cuda:0"
    n_classes = 10
    n_epoch = 20
    n_feat = 128
    n_T = 1000 # time steps
    LR = 1e-4

    now = datetime.datetime.now()
    save_dir = f'./pb1_output/mnistm_{now:%Y%m%d_%H%M}/'

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)
    ddpm.load_state_dict(torch.load("./pb1_output/epoch49_loss0.0160.pth"))
    # ddpm.load_state_dict(torch.load("./pb1_output/epoch21_loss0.0135.pth"))

    # tf = transforms.Compose([transforms.ToTensor()])
    dataset = MNISTM(f"./hw2_data/digits/mnistm")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.AdamW(ddpm.parameters(), lr=LR, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        optim.param_groups[0]['lr'] = LR * (1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            with torch.autocast(device_type='cuda' if device != 'cpu' else 'cpu', dtype=torch.float16):
                optim.zero_grad()
                x = x.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)
                loss = ddpm(x, c)
            scaler.scale(loss).backward()
            loss_ema = loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

        ddpm.eval()
        with torch.no_grad():
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # (1-2) 10 images for each digits
            x_gen = ddpm.sample(100, size=(3, 28, 28), device=device, guide_w=2.0)
            x_gen = x_gen.reshape(10, 10, 3, 28, 28)
            x_gen = torch.transpose(x_gen, 0, 1)
            x_gen = x_gen.reshape(-1, 3, 28, 28)
            save_image(x_gen, save_dir + f'ep{ep}_100_outputs.png', nrow=10)

            # all digits x100
            for idx in range(10):
                x_gen, x_gen_store = ddpm.sample_each(100, (3, 28, 28), device, idx, guide_w=2.0)
                for i in range(len(x_gen)):
                    save_image(x_gen[i], save_dir + f'{idx}_{i+1:03d}.png')
            # (1-3) '0; with six timesteps  
                    if idx == 0:
                        x_gen_store = x_gen_store.reshape(-1,3, 28, 28)
                        six_timestep = []
                        for k in range(len(x_gen_store)):
                            if k%100 == 9:
                                six_timestep.append(x_gen_store[k])
                        six_timestep_array = np.array(six_timestep)
                        six_timestep_tensor = torch.Tensor(six_timestep_array)
                        save_image(six_timestep_tensor, save_dir + f'six_timestep.png', nrow=len(x_gen_store))

        torch.save(ddpm.state_dict(), save_dir + f"ep{ep}_loss{loss_ema:.4f}.pth")
        print('saved model at ' + save_dir + f"ep{ep}_loss{loss_ema:.4f}.pth")

if __name__ == "__main__":
    train_mnist()

