import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils import beta_scheduler

# ref: https://github.com/xiaohu2015/nngen/blob/main/models/diffusion_models/ddim_mnist.ipynb
# ref: https://zhuanlan.zhihu.com/p/565698027

# def slerp(p0, p1, alpha):
#     dot = (p0 * p1).sum(dim=-1, keepdim=True)
#     dot = torch.clamp(dot, -1.0, 1.0)
#     theta = torch.acos(dot) * alpha
#     relative_p1 = p1 - p0 * dot
#     relative_p1 = relative_p1 / relative_p1.norm(dim=-1, keepdim=True)
#     return p0 * torch.cos(theta) + relative_p1 * torch.sin(theta)


class DDIM:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *( (1,) * ( len(x_shape) - 1)))
        return out

    @torch.no_grad()
    def sample(self, model, batch_size, channels, ddim_timesteps, ddim_eta, clip_denoised, noise):

        # make ddim timestep sequence
        c = self.timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        # sample_img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        noise = noise.reshape(1,3,256,256)
        sample_img = noise

        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)        

            # 2. predict noise using model
            pred_noise = model(sample_img, t)
            
            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
            # 6. compute x_{t-1} of formula (12)           
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)
            sample_img = x_prev

        return sample_img

    # def sample_slerp_linear(self, mode, specific_alpha, model, batch_size, channels, ddim_timesteps, ddim_eta, clip_denoised, noise):
    # def sample_slerp_linear(self, model, batch_size, channels, ddim_timesteps, ddim_eta, clip_denoised, noise):
        
    #     # make ddim timestep sequence
    #     c = self.timesteps // ddim_timesteps
    #     ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))

    #     # add one to get the final alpha values right (the ones from first scale to data during sampling)
    #     ddim_timestep_seq = ddim_timestep_seq + 1
    #     # previous sequence
    #     ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
    #     device = next(model.parameters()).device
    #     # start from pure noise (for each example in the batch)
    #     # sample_img = torch.randn((batch_size, channels, image_size, image_size), device=device)
    #     noise = noise.reshape(1,3,256,256)
    #     sample_img = noise
        
    #     # Allocate tensors outside the loop for memory reuse
    #     pred_dir_xt = torch.empty_like(sample_img)
    #     linear_interp_x_prev = torch.empty_like(sample_img)
    #     # alphas = specific_alpha * torch.ones(sample_img.shape, device=device)
    #     alphas = 0 * torch.ones(sample_img.shape, device=device)

    #     for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
    #         t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
    #         prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
    #         alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)

    #         # Get the current alpha_cumprod directly without tensor initialization
    #         alpha_cumprod_t = alphas

    #         pred_noise = model(sample_img, t)
            
    #         pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
    #         if clip_denoised:
    #             pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
    #         sigmas_t = ddim_eta * torch.sqrt(
    #             (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
    #         pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise

    #         # if mode == 'slerp':
                # slerped_x_prev = slerp(pred_x0, pred_dir_xt, alpha_cumprod_t)
                # x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + slerped_x_prev + sigmas_t * torch.randn_like(sample_img)
    #         # else:
    #         linear_interp_x_prev = (1 - alpha_cumprod_t) * pred_x0 + alpha_cumprod_t * pred_dir_xt
    #         x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + linear_interp_x_prev + sigmas_t * torch.randn_like(sample_img)

    #         sample_img = x_prev

    #     return sample_img

