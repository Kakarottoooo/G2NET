#!/usr/bin/env python
# coding: utf-8

# # Basic spectrogram image classification with Basic Audio Data Augmentation

# Code modified from JUN KODA's [Basic spectrogram image classification](https://www.kaggle.com/code/junkoda/basic-spectrogram-image-classification).
# In addition, @myso1987 introduced some basic audio data augmentations.

# In[1]:

# In[2]:


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import h5py
import timm
import torch
import torch.nn as nn
import torchaudio
import torchvision.transforms as TF


from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from timm.scheduler import CosineLRScheduler

device = torch.device('cuda')
criterion = nn.BCEWithLogitsLoss()

# Train metadata
di = '../input/g2net-detecting-continuous-gravitational-waves'
df = pd.read_csv(di + '/train_labels.csv')
df = df[df.target >= 0]  # Remove 3 unknowns (target = -1)


# # Dataset

# In[4]:


transforms_time_mask = nn.Sequential(
                torchaudio.transforms.TimeMasking(time_mask_param=10),
            )

transforms_freq_mask = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
            )

flip_rate = 0.0 # probability of applying the horizontal flip and vertical flip 
fre_shift_rate = 0.0 # probability of applying the vertical shift

time_mask_num = 0 # number of time masking
freq_mask_num = 0 # number of frequency masking


# In[5]:


def img_clipping(x, p=3, standard=256):
    point = np.percentile(x, 100-p)
    imgo = ((standard//2)*(x-x.min())/(point-x.min()))
    imgo = np.clip(imgo, 0, standard)
    return imgo.astype(np.float32)


# In[6]:


class Dataset(torch.utils.data.Dataset):
    """
    dataset = Dataset(data_type, df)

    img, y = dataset[i]
      img (np.float32): 2 x 360 x 128
      y (np.float32): label 0 or 1
    """
    def __init__(self, data_type, df, tfms=False):
        self.data_type = data_type
        self.df = df
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        """
        i (int): get ith data
        """
        r = self.df.iloc[i]
        y = np.float32(r.target)
        file_id = r.id

        img = np.empty((2, 360, 128), dtype=np.float32)

        filename = '%s/%s/%s.hdf5' % (di, self.data_type, file_id)
        with h5py.File(filename, 'r') as f:
            g = f[file_id]
            
            for ch, s in enumerate(['H1', 'L1']):
                a = g[s]['SFTs'][:, :4096] * 1e22  # Fourier coefficient complex64
                p = a.real**2 + a.imag**2  # power
                #p /= np.mean(p)  # normalize
                p -= np.mean(p)
                p = np.mean(p.reshape(360, 128, 32), axis=2)  # compress 4096 -> 128
                p = img_clipping(p)
                img[ch] = p

        if self.tfms:
            if np.random.rand() <= flip_rate: # horizontal flip
                img = np.flip(img, axis=1).copy()
            if np.random.rand() <= flip_rate: # vertical flip
                img = np.flip(img, axis=2).copy()
            if np.random.rand() <= fre_shift_rate: # vertical shift
                img = np.roll(img, np.random.randint(low=0, high=img.shape[1]), axis=1)
            
            img = torch.from_numpy(img)

            for _ in range(time_mask_num): # tima masking
                img = transforms_time_mask(img)
            for _ in range(freq_mask_num): # frequency masking
                img = transforms_freq_mask(img)
        
        else:
            img = torch.from_numpy(img)
                
        return img, y


# # Audio Data Augmentation

# * horizontal flip
# * vertical flip
# * vertical shift
# * time masking*
# * frequency masking*
# 
# *Reference  
# SpecAugment  
# https://arxiv.org/abs/1904.08779

# ## Horizontal flip and Vertical flip 

# In[7]:


dataset = Dataset('train', df, tfms=False)
img, y = dataset[10]


plt.figure(figsize=(8, 3))
plt.title('Spectrogram')
plt.xlabel('time')
plt.ylabel('frequency')
plt.imshow(img[0, 0:360])
plt.colorbar()
plt.show()


flip_rate = 1.0 # probability of applying the horizontal flip and vertical flip 

dataset = Dataset('train', df, tfms=True)
img, y = dataset[10]

plt.figure(figsize=(8, 3))
plt.title('Spectrogram')
plt.xlabel('time')
plt.ylabel('frequency')
plt.imshow(img[0, 0:360])
plt.colorbar()
plt.show()


# ## Vertical shift

# In[8]:


dataset = Dataset('train', df, tfms=False)
img, y = dataset[10]


plt.figure(figsize=(8, 3))
plt.title('Spectrogram')
plt.xlabel('time')
plt.ylabel('frequency')
plt.imshow(img[0, 0:360])
plt.colorbar()
plt.show()


flip_rate = 0.0 # probability of applying the horizontal flip and vertical flip 
fre_shift_rate = 1.0 # probability of applying the vertical shift

dataset = Dataset('train', df, tfms=True)
img, y = dataset[10]

plt.figure(figsize=(8, 3))
plt.title('Spectrogram')
plt.xlabel('time')
plt.ylabel('frequency')
plt.imshow(img[0, 0:360])
plt.colorbar()
plt.show()


# ## Time masking

# In[9]:


dataset = Dataset('train', df, tfms=False)
img, y = dataset[10]


plt.figure(figsize=(8, 3))
plt.title('Spectrogram')
plt.xlabel('time')
plt.ylabel('frequency')
plt.imshow(img[0, 0:360])
plt.colorbar()
plt.show()


flip_rate = 0.0 # probability of applying the horizontal flip and vertical flip 
fre_shift_rate = 0.0 # probability of applying the vertical shift
time_mask_num = 3 # number of time masking

dataset = Dataset('train', df, tfms=True)
img, y = dataset[10]

plt.figure(figsize=(8, 3))
plt.title('Spectrogram')
plt.xlabel('time')
plt.ylabel('frequency')
plt.imshow(img[0, 0:360])
plt.colorbar()
plt.show()


# ## Frequency masking

# In[10]:


dataset = Dataset('train', df, tfms=False)
img, y = dataset[10]


plt.figure(figsize=(8, 3))
plt.title('Spectrogram')
plt.xlabel('time')
plt.ylabel('frequency')
plt.imshow(img[0, 0:360])
plt.colorbar()
plt.show()


flip_rate = 0.0 # probability of applying the horizontal flip and vertical flip 
fre_shift_rate = 0.0 # probability of applying the vertical shift
time_mask_num = 0 # number of time masking
freq_mask_num = 3 # number of frequency masking

dataset = Dataset('train', df, tfms=True)
img, y = dataset[10]

plt.figure(figsize=(8, 3))
plt.title('Spectrogram')
plt.xlabel('time')
plt.ylabel('frequency')
plt.imshow(img[0, 0:360])
plt.colorbar()
plt.show()


# # Model

# In[11]:


class Model(nn.Module):
    def __init__(self, name, *, pretrained=False):
        """
        name (str): timm model name, e.g. tf_efficientnet_b2_ns
        """
        super().__init__()

        # Use timm
        model = timm.create_model(name, pretrained=pretrained, in_chans=2)

        clsf = model.default_cfg['classifier']
        n_features = model._modules[clsf].in_features
        model._modules[clsf] = nn.Identity()

        self.fc = nn.Linear(n_features, 1)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


# # Predict and evaluate

# In[12]:


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

    if pbar is not None:
        pbar = tqdm(desc='Predict', nrows=78, total=pbar)

    for img, y in loader_val:
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

        if pbar is not None:
            pbar.update(len(img))
        
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


# # Train

# In[13]:


model_name = 'tf_efficientnet_b7_ns'

nfold = 10
kfold = KFold(n_splits=nfold, random_state=42, shuffle=True)

epochs = 25
batch_size = 16
num_workers = 2
weight_decay = 1e-6
max_grad_norm = 1000

lr_max = 4e-4
epochs_warmup = 1.0


## setting of audio data augmentation 
flip_rate = 0.6 # probability of applying the horizontal flip and vertical flip 
fre_shift_rate = 0.6 # probability of applying the vertical shift
time_mask_num = 2 # number of time masking
freq_mask_num = 2 # number of frequency masking

for ifold, (idx_train, idx_test) in enumerate(kfold.split(df)):
    print('Fold %d/%d' % (ifold, nfold))
    torch.manual_seed(42 + ifold + 1)

    # Train - val split
    dataset_train = Dataset('train', df.iloc[idx_train], tfms=True)
    dataset_val = Dataset('train', df.iloc[idx_test])

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                     num_workers=num_workers, pin_memory=True, shuffle=True, drop_last=True)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                     num_workers=num_workers, pin_memory=True)

    # Model and optimizer
    model = Model(model_name, pretrained=True)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)

    # Learning-rate schedule
    nbatch = len(loader_train)
    warmup = epochs_warmup * nbatch  # number of warmup steps
    nsteps = epochs * nbatch        # number of total steps

    scheduler = CosineLRScheduler(optimizer,
                  warmup_t=warmup, warmup_lr_init=0.0, warmup_prefix=True, # 1 epoch of warmup
                  t_initial=(nsteps - warmup), lr_min=1e-6)                # 3 epochs of cosine
    
    time_val = 0.0
    lrs = []

    tb = time.time()
    print('Epoch   loss          score   lr')
    for iepoch in range(epochs):
        loss_sum = 0.0
        n_sum = 0

        # Train
        for ibatch, (img, y) in enumerate(loader_train):
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
            
            scheduler.step(iepoch * nbatch + ibatch + 1)
            lrs.append(optimizer.param_groups[0]['lr'])            

        # Evaluate
        val = evaluate(model, loader_val)
        time_val += val['time']
        loss_train = loss_sum / n_sum
        lr_now = optimizer.param_groups[0]['lr']
        dt = (time.time() - tb) / 60
        print('Epoch %d %.4f %.4f %.4f  %.2e  %.2f min' %
              (iepoch + 1, loss_train, val['loss'], val['score'], lr_now, dt))

    dt = time.time() - tb
    print('Training done %.2f min total, %.2f min val' % (dt / 60, time_val / 60))

    # Save model
    ofilename = 'model%d.pytorch' % ifold
    torch.save(model.state_dict(), ofilename)
    print(ofilename, 'written')

    break  # 1 fold only


# In[14]:


plt.title('LR Schedule: Cosine with linear warmup')
plt.xlabel('steps')
plt.ylabel('learning rate')
plt.plot(lrs)
plt.show()


# # Predict and submit

# In[15]:


# Load model (if necessary)
model = Model(model_name, pretrained=False)
filename = 'model0.pytorch'
model.to(device)
model.load_state_dict(torch.load(filename, map_location=device))
model.eval()

# Predict
submit = pd.read_csv(di + '/sample_submission.csv')
dataset_test = Dataset('test', submit)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64,
                                        num_workers=num_workers, pin_memory=True)

test = evaluate(model, loader_test, compute_score=False, pbar=len(submit))

# Write prediction
submit['target'] = test['y_pred']
submit.to_csv('submission2.csv', index=False)
print('target range [%.2f, %.2f]' % (submit['target'].min(), submit['target'].max()))

