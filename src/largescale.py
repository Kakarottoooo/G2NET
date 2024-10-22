import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import h5py
from scipy.stats import norm
import timm
import torch
import torch.nn as nn
import random
import os
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from timm.scheduler import CosineLRScheduler, PlateauLRScheduler
import pathlib
import sys
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2

def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Configurations for G2Net Training')
parser.add_argument('--seed', type=int, default=42, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--model', type=str, choices=['tf_efficientnetv2_b0'], default='tf_efficientnetv2_b0', 
                    help='type of model (default: tf_efficientnetv2_b0')                   
parser.add_argument('--weight', type=str, default='./checkpoints/tf_efficientnetv2_b0-c7cc451f.pth', 
                    help='weight path')            
parser.add_argument('--max_w', type=int, default=32, 
                    help='param for cutout')         
parser.add_argument('--max_h', type=int, default=32, 
                    help='param for cutout')    
parser.add_argument('--min_w', type=int, default=8, 
                    help='param for cutout')         
parser.add_argument('--min_h', type=int, default=8, 
                    help='param for cutout')             
parser.add_argument('--num_holes', type=int, default=128, 
                    help='param for cutout')          
parser.add_argument('--cut_p', type=float, default=0.8, 
                    help='param for cutout')   
args = parser.parse_args()

device = torch.device('cuda')
criterion = nn.BCEWithLogitsLoss()

# Train metadata
df = pd.read_csv('dataset/g2net2022/train_labels.csv')
df = df[df.target >= 0]  # Remove 3 unknowns (target = -1)

model_name = args.model
nfold = 5
folds = [0,1,2,3,4]
seed = args.seed
kfold = KFold(n_splits=nfold, random_state=seed, shuffle=True)
# kfold = StratifiedKFold(n_splits=nfold, random_state=seed, shuffle=True)

channels = 3
time_stamp = 64

scheduler_type = 'cos_once' # cos cos_once pleatu

epoches_each_round = 12 # when 'cos'
epochs_warmup = 4
epochs = 50 # epoches_each_round*5+epochs_warmup if 'cos'
decay_rate=0.2 # when pleatu
patience_t=5 # when pleatu

if_lookahead = False

batch_size = 32
num_workers = 2
weight_decay = 1e-6
max_grad_norm = 5
shuffle = True

lr_max = 1e-3

# base_dir = f'./result/1225_baseline_debug/cut{args.num_holes}_{args.max_h}_{args.min_h}_{args.max_w}_{args.min_w}_p{args.cut_p}_{model_name}_lr{lr_max}_e{epochs}_seed{seed}'
base_dir = f'./result/1225_reverse_debug/nocutnoshuffle_lr{lr_max}_gradnorm{max_grad_norm}'
multi_gpu = True
device_ids = [0,1,2,3]

pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True) 

work_dirs = [f'{base_dir}/{i}' for i in folds]
for work_dir in work_dirs:
    pathlib.Path(work_dir).mkdir(parents=True, exist_ok=True) 

cur_file_name = sys.argv[0].split('/')[-1]
shutil.copy(cur_file_name, f'{base_dir}/{cur_file_name}')  ## save the current script

set_seed(seed)

transform = {'train': A.Compose([
                    # A.CoarseDropout(max_holes = args.num_holes, max_width= args.max_w, 
                    #     max_height= args.max_h, min_height=args.min_h, 
                    #     min_width=args.min_w, fill_value= 0, p=args.cut_p),
                    # A.OneOf([
                    #     A.GridDistortion(num_steps=5, distort_limit=0.1, p=1),
                    #     A.ElasticTransform(p=1.0, alpha=0.8, sigma=0.5, alpha_affine=0.5)], p=0.7),
                    A.OneOf([
                        A.MedianBlur(blur_limit=3, p=1.),
                        A.MotionBlur(p=1.),
                        A.GaussianBlur(p=1),
                    ], p=0.7),
                    A.HorizontalFlip(p =0.5),
                    A.VerticalFlip(p=0.5),
                    ToTensorV2(p=1.),
                    ],
                    p=1),
            'test': A.Compose([
                                ToTensorV2(p=1.)
                                ], p=1)
                }

class Dataset(torch.utils.data.Dataset):
    """
    dataset = Dataset(data_type, df)

    img, y = dataset[i]
      img (np.float32): 2 x 360 x 128
      y (np.float32): label 0 or 1
    """
    def __init__(self, data_type, df, transform=None, shuffle=True):
        self.data_type = data_type
        self.df = df
        self.transform = transform
        self.shuffle = shuffle

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        """
        i (int): get ith data
        """
        r = self.df.iloc[i]
        y = np.float32(r.target)
        file_id = r.id

        fid, input, H1, L1 = dataload(f'./train_dataset/data/{file_id}.npy')
        img = preprocess(input, H1, L1).squeeze().numpy()
        img = np.transpose(img, (1,2,0))
        # img = np.concatenate([img, np.mean(img, axis=2, keepdims=True)], axis=2)
        if self.transform is not None: 
            img = self.transform(image=img)['image']
        # if self.data_type=='train' and self.shuffle: img = img[torch.randperm(img.size(0))]
        return img, y

def normalize(X):
    X = (X[..., None].view(X.real.dtype) ** 2).sum(-1)
    POS = int(X.size * 0.99903)
    EXP = norm.ppf((POS + 0.4) / (X.size + 0.215))
    scale = np.partition(X.flatten(), POS, -1)[POS]
    X /= scale / EXP.astype(scale.dtype) ** 2
    return X

df_meta = pd.read_csv('train_dataset/meta.csv')

def dataload(filepath):
    astime = np.load(filepath)
    fid = filepath.split('/')[-1].split('.')[0]
    H1 = df_meta[df_meta['fid']==fid]['H1'].values[0]
    L1 = df_meta[df_meta['fid']==fid]['L1'].values[0]
    return fid, astime, H1, L1
class LargeKernel_debias(nn.Conv2d):
    def forward(self, input: torch.Tensor):
        finput = input.flatten(0, 1)[:, None]
        target = abs(self.weight)
        target = target / target.sum((-1, -2), True)
        joined_kernel = torch.cat([self.weight, target], 0)
        reals = target.new_zeros(
            [1, 1] + [s + p * 2 for p, s in zip(self.padding, input.shape[-2:])]
        )
        reals[
            [slice(None)] * 2 + [slice(p, -p) if p != 0 else slice(None) for p in self.padding]
        ].fill_(1)
        output, power = torch.nn.functional.conv2d(
            finput, joined_kernel, padding=self.padding
        ).chunk(2, 1)
        ratio = torch.div(*torch.nn.functional.conv2d(reals, joined_kernel).chunk(2, 1))
        power_ = power.mul(ratio)
        output_ = output.sub(power_)
        return output_.unflatten(0, input.shape[:2]).flatten(1, 2)

def preprocess(input, H1, L1):
    # input = torch.from_numpy(input).to("cuda", non_blocking=True)
    # rescale = torch.tensor([[H1, L1]]).to("cuda", non_blocking=True)
    input = torch.from_numpy(input)
    rescale = torch.tensor([[H1, L1]])
    tta = (
        torch.randn(
            [1, *input.shape, 2], device=input.device, dtype=torch.float32
        )
        .square_()
        .sum(-1)
    )
    tta *= rescale[..., None, None] / 2
    valid = ~torch.isnan(input); tta[:, valid] = input[valid].float()
    return tta

def get_model(path):
    model = timm.create_model(
        model_name,
        in_chans=32,
        num_classes=1,
    )
    state_dict = torch.load(path)
    C, _, H, W = 16, 1, 31, 255
    
    model.conv_stem = nn.Sequential(
        nn.Identity(),
        nn.AvgPool2d((1, 9), (1, 8), (0, 4), count_include_pad=False),
        LargeKernel_debias(1, C, [H, W], 1, [H//2, W//2], 1, 1, False),
        model.conv_stem,
    )
    state_dict.pop("classifier.weight")
    state_dict.pop("classifier.bias")
    model.load_state_dict(state_dict, strict=False)
    model.cuda().eval()
    return model


def evaluate(model, loader_val, *, compute_score=True, pbar=None):
    """
    Predict and compute loss and score
    """
    tb = time.time()
    was_training = model.training
    model.eval()

    loss_sum = 0.0
    n_sum = 0
    y_all = []
    y_pred_all = []

    pbar = tqdm(enumerate(loader_val),total = len(loader_val),desc = 'Predict')
    for step, (img, y) in pbar:
        n = y.size(0)
        img = img.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = model(img.to(device))
        loss = criterion(y_pred.view(-1), y)

        n_sum += n
        loss_sum += n * loss.item()

        y_all.append(y.cpu().detach().numpy())
        y_pred_all.append(y_pred.sigmoid().squeeze().cpu().detach().numpy())

        
        del loss, y_pred, img, y

    loss_val = loss_sum / n_sum

    y = np.concatenate(y_all)
    y_pred = np.concatenate(y_pred_all)

    score = roc_auc_score(y, y_pred) if compute_score else None

    ret = {'loss': loss_val,
           'score': score,
           'y': y,
           'y_pred': y_pred,
           'time': time.time() - tb}
    
    model.train(was_training)  # back to train from eval if necessary

    return ret

LOGGER = init_logger(log_file = f'{base_dir}/train.log')
for ifold, (idx_train, idx_test) in enumerate(kfold.split(df)):
# for ifold, (idx_train, idx_test) in enumerate(kfold.split(df, df['target'].values)):
    if ifold not in folds: continue
    work_dir = f'{base_dir}/{ifold}/'
    
    LOGGER.info('**********************Fold %d/%d***************************' % (ifold, nfold))
    # torch.manual_seed(42 + ifold + 1)

    # Train - val split
    dataset_train = Dataset('train', df.iloc[idx_train], transform=transform['train'], shuffle=True)
    dataset_val = Dataset('valid', df.iloc[idx_test], transform=transform['test'], shuffle=False)

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                     num_workers=num_workers, pin_memory=False, shuffle=True, drop_last=True)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                     num_workers=num_workers, pin_memory=False)

    # Model and optimizer
    model = get_model(args.weight).cuda()
    # state_dict = torch.load(args.weight,map_location=torch.device('cuda:0'))
    # model.load_state_dict(state_dict)
    if multi_gpu: model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)

    # Learning-rate schedule
    nbatch = len(loader_train)
    warmup = epochs_warmup * nbatch  # number of warmup steps
    initials_nsteps = epoches_each_round * nbatch        # number of total steps
    if scheduler_type == 'cos':
        scheduler = CosineLRScheduler(optimizer,
                    warmup_t=warmup, warmup_lr_init=0.0, warmup_prefix=True, # epoch of warmup
                    t_initial=initials_nsteps, lr_min=1e-6)                # epochs of the first cosine round
    if scheduler_type == 'cos_once':
        scheduler = CosineLRScheduler(optimizer,
                    warmup_t=warmup, warmup_lr_init=0.0, warmup_prefix=True, # epoch of warmup
                    t_initial=epochs*nbatch -warmup, lr_min=1e-6)                # epochs of the first cosine round
    
    time_val = 0.0
    lrs = []

    tb = time.time()
    LOGGER.info('Epoch   loss          score   lr')
    best_score, best_loss = 0, 100
    for iepoch in range(epochs):
        loss_sum = 0.0
        n_sum = 0
        pbar = tqdm(enumerate(loader_train),total = len(loader_train),desc = 'Train')
        # Train
        for ibatch, (img, y) in pbar:
  
            n = y.size(0)
            img = img.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_pred = model(img)
            loss = criterion(y_pred.view(-1), y)

            loss_train = loss.item()
            loss_sum += n * loss_train
            n_sum += n

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_grad_norm)
            optimizer.step()
            
            if scheduler_type != 'pleatu': scheduler.step(iepoch * nbatch + ibatch + 1)
            lrs.append(optimizer.param_groups[0]['lr'])            

        # Evaluate
        val = evaluate(model, loader_val)
    
        time_val += val['time']
        loss_train = loss_sum / n_sum
        lr_now = optimizer.param_groups[0]['lr']
        dt = (time.time() - tb) / 60
        val_loss, val_score = val['loss'], val['score']
        LOGGER.info(f'Epoch {iepoch + 1} {round(loss_train, 4)} {round(val_loss, 4)} {round(val_score, 4)} {lr_now} {round(dt, 2)} min')

        if scheduler_type == 'pleatu': scheduler.step(val_score)

        # Save model
        if val['score'] > best_score:
            LOGGER.info(f'Best score improves from {best_score} to {val_score}')
            best_score = val['score']
            ofilename = f'{work_dir}/best_score.pytorch'
            torch.save(model.state_dict(), ofilename)
            
        if val['loss'] < best_loss:
            LOGGER.info(f'Best loss improves from {best_loss} to {val_loss}')
            best_loss = val['loss']
            ofilename = f'{work_dir}/best_loss.pytorch'
            torch.save(model.state_dict(), ofilename)         

    dt = time.time() - tb
    LOGGER.info(f'Training done {round(dt/60, 2)} min total, {round(time_val/60, 2)} min val')
    # LOGGER.shutdown()




