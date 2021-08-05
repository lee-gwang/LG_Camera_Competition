# - train.py - #

# ------------------------
#  Import library
# ------------------------
import numpy as np
import os, sys
import pandas as pd
import cv2
import argparse
import segmentation_models_pytorch as smp
import torch
import torch.cuda.amp as amp
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR, OneCycleLR
from warmup_scheduler import GradualWarmupScheduler
from adamp import AdamP
from torch.optim.optimizer import Optimizer
import tqdm
import random
from sklearn.model_selection import KFold
#
from dataloader import LGDataSet, get_transforms
from models.model import SRModels
from utils import *

# ------------------------
#  Arguments
# ------------------------
parser = argparse.ArgumentParser(description='LG')
# setting
parser.add_argument('--debug', action='store_true', help='debugging mode')
parser.add_argument('--amp', type=bool, default=True, help='mixed precision')
parser.add_argument('--gpu', type=str, default= '0,1', help='gpu')
parser.add_argument('--img_size', type=int, default= 512, help='image size/ [512, 768]')
# training
parser.add_argument('--scheduler', type=str, default= 'warmupv2', help='')
parser.add_argument('--epochs', type=int, default= 50, help='')
parser.add_argument('--start_lr', type=float, default= 3e-5, help='start learning rate')
parser.add_argument('--warmup_epo', type=int, default= 1, help='')
parser.add_argument('--cosine_epo', type=int, default= 49, help='')
parser.add_argument('--warmup_factor', type=int, default= 10, help='')
parser.add_argument('--batch_size', type=int, default= 32, help='')
parser.add_argument('--weight_decay', type=float, default= 1e-6, help='')
parser.add_argument('--alpha', type=float, default= 0.1, help='loss ratio')
parser.add_argument('--loss', type=str, default= 'mae+mse', help='')
# model
parser.add_argument('--activation', type=str, default= 'sigmoid', help='')
parser.add_argument('--encoder', type=str, default= 'se_resnext50_32x4d', help='resnet34, ,,,')
parser.add_argument('--decoder',  type=str, default= 'unet', help='unet, fpn...')
# else
parser.add_argument('--exp_name', type=str, default= 'experiment', help='experiments name')
parser.add_argument('--num_workers', type=int, default= 8, help='')
parser.add_argument('--seed', type=int, default= 42, help='')

args = parser.parse_args()

args.dir_ = f'./saved_models/{args.encoder}_{args.img_size}_{args.decoder}_{args.exp_name}/'

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # for faster training, but not deterministic


# ------------------------
#  scheduler
# ------------------------
def get_scheduler(optimizer):
    if args.scheduler=='ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, 
                                      min_lr = 1e-5, verbose=True, eps=args.eps)
    elif args.scheduler=='CosineAnnealingLR':
        print('scheduler : Cosineannealinglr')
        scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr, last_epoch=-1)
    elif args.scheduler=='CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=1, eta_min=args.min_lr, last_epoch=-1)
    elif args.scheduler == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=args.decay_epoch, gamma= args.factor, verbose=True)
    elif args.scheduler == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                      max_lr=1e-3, epochs=args.epochs, steps_per_epoch=len(train_loader))
        
    elif args.scheduler =='warmupv2':
        scheduler_cosine=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.cosine_epo)
        scheduler_warmup=GradualWarmupSchedulerV2(optimizer, multiplier=args.warmup_factor, total_epoch=args.warmup_epo, after_scheduler=scheduler_cosine)
        scheduler=scheduler_warmup  
    else:
        scheduler = None
        print('scheduler is None')
    return scheduler


# ------------------------
#  Train
# ------------------------
def run_train():
    # ------------------------
    # Logger
    # ------------------------
    out_dir = args.dir_
    log = Logger()
    os.makedirs(out_dir, exist_ok=True)
    log.open(out_dir + '/log.train.txt', mode='a')
    print_args(args, log)
    log.write('\n')
    
    # ------------------------
    # Dataloader
    # ------------------------
    
    # load preprocess data
    df = pd.read_csv(f'./data/preprocess_train_{args.img_size}.csv')
    log.write('load dataset'+'\n')
    
    # debut
    if args.debug:
        df = df.sample(5000).copy()
    
    # trainloader
    train_data = df[df['type_']=='train'].reset_index(drop=True).copy()

    train_transform = get_transforms(data='train', img_size=args.img_size)
    train_dataset = LGDataSet(data = train_data, transform = train_transform)
    trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                             num_workers=8, shuffle=True, pin_memory=True)

    # ------------------------
    # Model
    # ------------------------
    scaler = amp.GradScaler()
    net = SRModels()

    net.to(device)
    if len(args.gpu)>1:
        net = nn.DataParallel(net)

    # ------------------------
    # loss
    # ------------------------
    if args.loss =='mae':
        loss_fn = nn.L1Loss()
    elif args.loss =='mse':
        loss_fn = nn.MSELoss()
    elif args.loss =='mae+mse':
        loss_fn = nn.L1Loss()
        loss_fn2 = nn.MSELoss()

    # ------------------------
    #  Optimizer
    # ------------------------
    optimizer = AdamP(net.parameters(), lr=args.start_lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer)

    # ------------------------
    #  Training Start
    # ------------------------
    best_score = 0
    for epoch in range(1, args.epochs+1):
        train_loss = 0
        valid_loss = 0

        target_lst = []
        pred_lst = []
        lr = get_learning_rate(optimizer)
        log.write(f'-------------------')
        log.write(f'{epoch}epoch start')
        log.write(f'-------------------'+'\n')
        #log.write(f'learning rate : {lr : .6f}')
        for t, (images, targets) in enumerate(tqdm.tqdm(trainloader)):
            images  = images.to(device=device, dtype=torch.float)
            targets = targets.to(device=device, dtype=torch.float)

            net.train()
            optimizer.zero_grad()
            
            # mixed precision (fp16)
            if args.amp:
                with amp.autocast():
                    output = net(images)
                    if args.loss == 'mae+mse':
                        loss = args.alpha *loss_fn(output, targets) +(1-args.alpha)* loss_fn2(output, targets)
                    else :
                        loss = loss_fn(output, targets)
                    train_loss += loss


                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # normal training (fp32)
            else:
                output = net(images)
                if args.loss == 'mae+mse':
                    loss = args.alpha *loss_fn(output, targets) +(1-args.alpha)* loss_fn2(output, targets)
                else :
                    loss = loss_fn(output, targets)
                train_loss += loss

                # update
                loss.backward()
                optimizer.step()

        if scheduler is not None:
            scheduler.step() 
        train_loss = train_loss / len(trainloader)

        # validation
        valid_score, psnr_score_list = do_psnr(net)

        if valid_score > best_score:
            best_score = valid_score
            best_epoch = epoch
            best_loss = valid_loss

            torch.save(net.state_dict(), out_dir + f'/{epoch}e_{best_score:.4f}_s.pth')
            log.write('best model saved'+'\n')


        log.write(f'train loss : {train_loss:.4f}'+'\n')
        log.write(f'valid score :{valid_score:.2f} '+'\n')
        log.write(f'{[str(round(x, 2)) for x in psnr_score_list]}'+'\n')

# ------------------------
#  Validation
# ------------------------
def do_psnr(net):
    df = pd.read_csv(f'./data/preprocess_train_{args.img_size}.csv')

    list_ = df[df['type_']=='val']['img_id'].unique().tolist()
    img_paths =[] ; label_paths = []
    for id_ in list_:
        img_paths.append(f'./data/train/train_input_{id_}.png')
        label_paths.append(f'./data/train/train_label_{id_}.png')

    
    img_size = args.img_size ; stride = args.img_size//2

    batch_size = 32
    results = []
    psnr_score_list = []
    net.eval()
    with torch.no_grad():
        if args.amp:
            with amp.autocast():
                for img_path in tqdm.tqdm(img_paths):
                    img = cv2.imread(img_path)
                    img = img.astype(np.float32)/255
                    crop = []
                    position = []
                    batch_count = 0

                    result_img = np.zeros_like(img)
                    voting_mask = np.zeros_like(img)

                    for top in range(0, img.shape[0], stride):
                        for left in range(0, img.shape[1], stride):
                            piece = np.zeros([img_size, img_size, 3], np.float32)
                            temp = img[top:top+img_size, left:left+img_size, :]
                            piece[:temp.shape[0], :temp.shape[1], :] = temp
                            crop.append(piece)
                            position.append([top, left])
                            batch_count += 1
                            
                            if batch_count == batch_size:
                                crop = torch.tensor(np.array(crop)).permute(0,3,1,2).to(device)

                                pred = net(crop)
                                pred = pred.detach().cpu().numpy()*255
                                crop = []
                                batch_count = 0
                                for num, (t, l) in enumerate(position):
                                    piece = pred[num]
                                    h, w, c = result_img[t:t+img_size, l:l+img_size, :].shape
                                    result_img[t:t+img_size, l:l+img_size, :] += piece[:h, :w]
                                    voting_mask[t:t+img_size, l:l+img_size, :] += 1
                                position = []
                    if batch_count != 0: 
                        crop = torch.tensor(np.array(crop)).permute(0,3,1,2).to(device)
                        pred = net(crop)*255
                        pred = pred.detach().cpu().numpy()
                        crop = []
                        batch_count = 0
                        for num, (t, l) in enumerate(position):
                            piece = pred[num]
                            h, w, c = result_img[t:t+img_size, l:l+img_size, :].shape
                            result_img[t:t+h, l:l+w, :] += piece[:h, :w]
                            voting_mask[t:t+h, l:l+w, :] += 1
                        position = []
                        
                    result_img = result_img/voting_mask
                    result_img = np.around(result_img).astype(np.uint8)
                    results.append(result_img)
    
    for i, (input_path, label_path) in enumerate(zip(img_paths, label_paths)):
        
        targ_img = cv2.imread(label_path)
        psnr_score_list.append(psnr_score(results[i].astype(float), targ_img.astype(float), 255))
        
    
    return np.mean(psnr_score_list), psnr_score_list

if __name__ == '__main__':
    set_seeds(seed=args.seed) 
    run_train()