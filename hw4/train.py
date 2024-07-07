import os, sys
from options import get_options
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
# from pytorch_lightning.logging import TestTubeLogger
# from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

        self.val_loss = []
        self.val_psnr = []

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'json_dir':self.hparams.json_dir}
                #   'img_wh': tuple(self.hparams.img_wh)}

        # if self.hparams.dataset_name == 'llff':
        #     kwargs['spheric_poses'] = self.hparams.spheric_poses
        #     kwargs['val_num'] = self.hparams.num_gpus

        self.train_dataset = dataset(split='train', **kwargs) # len = 5308416
        self.val_dataset = dataset(split='val', **kwargs)     # len = 20


    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        log['train/loss'] = loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            # self.logger.log_images('val/GT_pred_depth', stack, self.global_step)
            self.logger.experiment.add_images('val/GT_pred_depth', stack, self.global_step)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        self.val_loss.append(log['val_loss'].detach())
        self.val_psnr.append(log['val_psnr'].detach())

        return log

    # def validation_epoch_end(self, outputs):
        # mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
    def on_validation_epoch_end(self):
        # print('loss',self.val_loss)
        # print('psnr',self.val_psnr)
        mean_loss = torch.stack(self.val_loss).mean()
        mean_psnr = torch.stack(self.val_psnr).mean()
        print('mean_loss', mean_loss)
        print('mean_psnr', mean_psnr)

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
               }


if __name__ == '__main__':
    hparams = get_options()
    system = NeRFSystem(hparams)

    system.prepare_data() 
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'ckpts/{hparams.exp_name}',
        filename='{epoch:02d}_{val_psnr:02d}_{val_loss:.4f}',
        mode='min',
        )

    # logger = TestTubeLogger(save_dir="logs", name=hparams.exp_name, debug=False, create_git_tag=False)
    # logger = CSVLogger('logs', name=hparams.exp_name)
    logger = TensorBoardLogger('logs', name=hparams.exp_name)

    print()
    print('-----trainer-----')
    trainer = Trainer(accelerator = 'gpu',
                      devices = 1,
                      logger=logger,
                      callbacks=checkpoint_callback,
                      max_epochs=hparams.num_epochs,
                      benchmark=True,
                    )

    print()
    print('-----trainer.fit(system)-----')
    trainer.fit(system)
