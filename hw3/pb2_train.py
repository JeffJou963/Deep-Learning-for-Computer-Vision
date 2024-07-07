import os
import json
import math
import timm
import torch
import argparse
import datetime
import numpy as np
import torch.nn as nn
import loralib as lora

from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloader import pb2_dataset
from pb2_model import Transformer
from pb2_model_adapter import Transformer_adapter
from pb2_model_tuning import Transformer_tuning
from pb2_model_lora import Transformer_lora
from tokenizer import BPETokenizer
from p2_evaluate import eval

parser = argparse.ArgumentParser()
parser.add_argument('--type', default='lora')
parser.add_argument('--data_dir', default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/images/train')
parser.add_argument('--val_data_dir', default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/images/val')
parser.add_argument('--out_dir', default='./output_p2/pred.json')
parser.add_argument('--decoder_weights', default='/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/decoder_model.bin')
args = parser.parse_args()

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
now = datetime.datetime.now()

BATCHSIZE = 4
n_epochs = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# encoder_path = 'vit_giant_patch14_clip_224.laion2b'
# encoder_path = 'eva_giant_patch14_224.clip_ft_in1k'
encoder_path = 'vit_gigantic_patch14_clip_224'

if args.type == 'lora':
    model = Transformer_lora(encoder_path, args.decoder_weights) # 28965888, 29260800, 30735360, 30753792, 31695024
    model_path = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/CIDER0.8410.ckpt'

    lora.mark_only_lora_as_trainable(model)
    for name, param in model.decoder.named_parameters():
        if 'cross_attn' in name or 'ln_2' in name or 'linear' in name:
            param.requires_grad = True
    print('LORA')
else:
    if args.type == 'adapter':
        model = Transformer_adapter(encoder_path, args.decoder_weights) # 30,202,032
        print('ADAPTER')
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.decoder.named_parameters():
            if 'cross_attn' in name or 'adapter' in name:
                param.requires_grad = True

    elif args.type == 'tuning': # 28,348,416 28366848
        model = Transformer_tuning(encoder_path, args.decoder_weights)
        print('TUNING')
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.decoder.named_parameters():
            if 'cross_attn' in name or 'ln_2' in name:
                param.requires_grad = True

    else: 
        model = Transformer(encoder_path, args.decoder_weights) # 28,348,416 28366848
        model_path = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/28_20_23_lora_val_ep10_2.995630.ckpt'
        model.load_state_dict(torch.load(model_path, device), strict=False)
        print('else')
        for param in model.encoder.parameters(): # 153,888,768
            param.requires_grad = False
        # for param in model.parameters():
        #     param.requires_grad = False
        for name, param in model.decoder.named_parameters():
            if 'cross_attn' in name:
                param.requires_grad = True

trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
save_weights = {k: v for k,v in model.state_dict().items() if k in trainable_weights}
print('Total params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

encoder_file = './encoder.json'
vocab_file = './vocab.bpe'
tokenizer = BPETokenizer(encoder_file, vocab_file)

train_json_dir = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/train.json'
val_json_dir = '/home/remote/yichou/DLCV/dlcv-fall-2023-hw3-JeffJou963/hw3_data/p2_data/val.json'

train_dataset = pb2_dataset(args.data_dir, train_json_dir,'train')
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
val_dataset = pb2_dataset(args.val_data_dir, val_json_dir, 'val')
# val_dataset = pb2_dataset(args.val_data_dir, val_json_dir, 'test')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(params=params, lr=3e-5)
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=0)
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs*len(train_loader), eta_min=1e-8)
criterion = nn.CrossEntropyLoss(ignore_index=-100) 

# ---------- Training ----------
step, val_step = 0,0
best_val_loss=10
best_loss = 10

writer = SummaryWriter()
model = model.to(device)

for epoch in range(n_epochs):
    train_losses = []

    for update, (img_file_name, image, caption_in, caption_gt) in tqdm(enumerate(train_loader)):
        model.train()
        image = image.to(device)
        caption_in = caption_in.to(device)
        caption_gt = caption_gt.to(device)  # (b, max_cap_length=50)

        optimizer.zero_grad()
        with autocast():
            output = model(image, caption_in) # (b,max_cap_len,50257)
            # output.view(-1,50257)          #(b*max_cap_len,50257)
            loss = criterion(output.view(-1, output.size(-1)), caption_gt.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        if not math.isnan(loss.item()):
            train_losses.append(loss.item())
        else:
            print('nan',update)

        if update % 3000 == 2999:            
            step+=1
            train_loss_avg = sum(train_losses) / len(train_losses)
            writer.add_scalar("Loss/Train", train_loss_avg,step)
            print(f"[ Train | {epoch + 1:02d}/{n_epochs:02d} ] loss = {train_loss_avg:.4f}")
            train_losses = []

            # if train_loss_avg < best_loss:
            print(f"Best model found at epoch {epoch+1}, loss={train_loss_avg:4f}, SAVE")
            #     trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
            #     save_weights = {k: v for k,v in model.state_dict().items() if k in trainable_weights}
            #     print('Total params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
            #     torch.save(save_weights, f'{now.day}_{now.hour}_{now.minute}_{args.type}_ep{epoch+1}_{train_loss_avg:4f}.ckpt')
            #     best_loss = train_loss_avg
            # Predict & Calculate Score

            json_dict = {}
            model.eval()
            with torch.no_grad():            
                # for val_img_file_name, val_image in tqdm(val_loader):
                for (val_img_file_name,), val_image, val_caption_in, val_caption_gt in tqdm(val_loader):

                    val_image = val_image.to(device)
                    
                    encoder_out = model.encoder.forward_features(val_image)
                    total_pred=[]
                    cap_in = torch.full((1, 60), 50256)
                    for i in range(59):
                        cap_in = cap_in.to(device)
                        output = model.decoder(encoder_out, cap_in) #(1,60,50257)
                        output = torch.argmax(output, dim=-1) # (1,60)
                        current = output.squeeze(0)[i].item() # (1)
                        cap_in[0][i+1] = current             
                        # print('current',current)
                        if current == 50256:
                            break
                        else:
                            total_pred.append(current)
                    total_pred = tokenizer.decode(total_pred)
                    print(f'{val_img_file_name} : {total_pred}')
                    json_dict[val_img_file_name] = total_pred

                with open(args.out_dir, 'w') as json_file:
                    json.dump(json_dict, json_file, indent=2)

                cider_score, clip_score = eval(args.out_dir, args.val_data_dir, val_json_dir)
                print(f"CIDEr: {cider_score} | CLIPScore: {clip_score}")
                
                trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
                save_weights = {k: v for k,v in model.state_dict().items() if k in trainable_weights}
                torch.save(save_weights, f'CIDER{cider_score:.4f}_{now.hour}_{now.minute}_{args.type}_ep{epoch+1}_{train_loss_avg:4f}.ckpt')


            # # Validation
            # val_losses = []
            # json_dict = {}
            # model.eval()
            # with torch.no_grad():            
                # for (val_img_file_name,), val_image, val_caption_in, val_caption_gt in tqdm(val_loader):
            #         val_image = val_image.to(device)
            #         val_caption_in = val_caption_in.to(device)
            #         val_caption_gt = val_caption_gt.to(device)

            # # Calculate loss
            #         val_output = model(val_image, val_caption_in)
            #         val_loss = criterion(val_output.view(-1, val_output.size(-1)), val_caption_gt.view(-1))
            #         val_losses.append(val_loss.item())

            # val_loss_avg = sum(val_losses) / len(val_losses)
            # val_losses = []
            # val_step+=1

            # if val_loss_avg < best_val_loss:
            #     trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
            #     save_weights = {k: v for k,v in model.state_dict().items() if k in trainable_weights}
            #     torch.save(save_weights, f'ad&lo_{now.day}_{now.hour}_{now.minute}_{args.type}_val_ep{epoch+1}_{val_loss_avg:4f}.ckpt')
            #     best_val_loss = val_loss_avg

            #     print('Total params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
            #     print(f"Best model found at epoch {epoch+1}, loss={val_loss_avg:4f}, SAVE")
            