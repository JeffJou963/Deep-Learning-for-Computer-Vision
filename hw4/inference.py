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

from datasets import dataset_dict
from datasets.depth_utils import *

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--json_dir', type=str, default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw4-JeffJou963/dataset/')
    parser.add_argument('--output_dir', type=str, default='./output_inf')

    parser.add_argument('--ckpt_path', type=str, default='epoch=15_fine=192.ckpt')
    parser.add_argument('--scene_name', type=str, default='test_fine192', help='scene name, used as output folder name')

    parser.add_argument('--dataset_name', type=str, default='Klevr', help='which dataset to validate')
    parser.add_argument('--split', type=str, default='test', help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[256, 256], help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true", help='whether images are taken in spheric poses (for llff)')
    parser.add_argument('--N_samples', type=int, default=64, help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=192, help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true", help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*4, help='chunk size to split the input to avoid OOM')

    parser.add_argument('--save_depth', default=False, action="store_true", help='whether to save depth prediction')
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

if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    kwargs = {
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

    os.makedirs(args.output_dir, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        image_id = sample['image_id']
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back)

        img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()

        img_pred_ = (img_pred*255).astype(np.uint8) # float32 -> uint8
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(args.output_dir, f'{image_id:05d}.png'), img_pred_)
