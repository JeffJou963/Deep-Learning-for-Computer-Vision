import torch
import os
import numpy as np
import scipy.signal
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

from grade import render_viewpoints
from utils import *
import cv2

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--json_dir', type=str, default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw4-JeffJou963/dataset/metadata.json')
    parser.add_argument('--root_dir', type=str, default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw4-JeffJou963/dataset')
    parser.add_argument('--output_dir', type=str, default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw4-JeffJou963/output')

    # parser.add_argument('--ckpt_path', type=str, default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw4-JeffJou963/ckpts/exp1/epoch=15.ckpt')
    # parser.add_argument('--scene_name', type=str, default='val_test', help='scene name, used as output folder name')

    # parser.add_argument('--ckpt_path', type=str, default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw4-JeffJou963/ckpts/epoch=15_coarse=16.ckpt')
    # parser.add_argument('--scene_name', type=str, default='val_coarse16', help='scene name, used as output folder name')

    parser.add_argument('--ckpt_path', type=str, default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw4-JeffJou963/ckpts/epoch=15_fine=192.ckpt')
    parser.add_argument('--scene_name', type=str, default='val_fine192', help='scene name, used as output folder name')
    # parser.add_argument('--scene_name', type=str, default='test_fine192', help='scene name, used as output folder name')

    parser.add_argument('--dataset_name', type=str, default='Klevr', help='which dataset to validate')
    # parser.add_argument('--split', type=str, default='test', help='test or test_train')
    parser.add_argument('--split', type=str, default='val', help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[256, 256], help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true", help='whether images are taken in spheric poses (for llff)')
    parser.add_argument('--N_samples', type=int, default=64, help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=192, help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true", help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*4, help='chunk size to split the input to avoid OOM')

    parser.add_argument('--save_depth', default=True, action="store_true", help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm', choices=['pfm', 'bytes'], help='which format to save')

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk,
                      white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    chunk = 1024*32
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    kwargs = {'root_dir': args.root_dir,
              'json_dir': args.json_dir,
              'split': args.split,}
            #   'img_wh': tuple(args.img_wh)}
    dataset = dataset_dict[args.dataset_name](**kwargs)

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    nerf_coarse = NeRF()
    nerf_fine = NeRF()
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    nerf_coarse.cuda().eval()
    nerf_fine.cuda().eval()

    models = [nerf_coarse, nerf_fine]
    embeddings = [embedding_xyz, embedding_dir]
    
    device = "cuda"
    imgs = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    dir_name = f'{args.output_dir}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    for i in tqdm(range(len(dataset))):
    # for i, idx in tqdm(enumerate(dataset)):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back)

        # for key, value in results.items():
            # print(f"{key}': {value.shape}")
            # opacity_coarse': torch.Size([65536])
            # rgb_fine': torch.Size([65536, 3]) -> [256*256,3]
            # depth_fine': torch.Size([65536])
            # opacity_fine': torch.Size([65536])

        img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()

        if args.save_depth:
            depth_pred = results['depth_fine'].view(h, w).cpu().numpy()
            depth_pred = np.nan_to_num(depth_pred)

            img = visualize_depth(depth_pred, cmap=cv2.COLORMAP_JET)
            cv2.imwrite(f'{args.output_dir}/{args.dir_name}/{i:02d}.jpg', img)

            # if args.depth_format == 'pfm':
            #     save_pfm(os.path.join(dir_name, f'depth_{i:03d}.pfm'), depth_pred)
            # else:
            #     with open(f'depth_{i:03d}', 'wb') as f:
            #         f.write(depth_pred.tobytes())

        img_pred_ = (img_pred*255).astype(np.uint8) # float32 -> uint8
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        if 'rgbs' in sample:
            # for key, value in sample.items():
                # print(f'{key}: {value.shape}')
                # rays: torch.Size([65536, 8])
                # c2w: torch.Size([3, 4])
                # rgbs: torch.Size([65536, 3])
                # valid_mask: torch.Size([65536])

            rgbs = sample['rgbs'] 
            img_gt = rgbs.view(h, w, 3) #  torch.float32
            psnrs += [metrics.psnr(img_gt, img_pred).item()]

            img_gt = np.array(img_gt)
            ssims.append(rgb_ssim(img_pred, img_gt, max_val=1))
            lpips_alex.append(rgb_lpips(img_pred, img_gt, net_name='alex', device=device))
            lpips_vgg.append(rgb_lpips(img_pred, img_gt, net_name='vgg', device=device))

    # imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=30)
    
    if psnrs:
        print('psnr', np.mean(psnrs))
        print('ssim', np.mean(ssims))
        print('lpips (vgg)', np.mean(lpips_vgg))
        print('lpips (alex)', np.mean(lpips_alex))

        
    # psnrs_, ssims, lpips_vgg, lpips_alex = render_viewpoints(args.json_dir, args.output_dir)
    # print('psnrs_:', psnrs_)
    # print('ssims:', ssims)
    # print('lpips_vgg:', lpips_vgg)
    # print('lpips_alex:', lpips_alex)