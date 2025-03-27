import torch
import wandb
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset, DataLoader, Subset
# from preprocess.PreprocessorReg import videoToMask
from sklearn.model_selection import train_test_split
from src.utils.VideoDataset import VideoDataset
from src.model.ViscosityEstimator import ViscosityEstimator
from src.model.ViscosityResnet import ViscosityResnet
from src.losses.MSLE import MSLELoss
import os.path as osp
import glob
from statistics import mean
import yaml
import json

# Load Config
with open("config_reg.yaml", "r") as file:
    config = yaml.safe_load(file)

BATCH_SIZE = int(config["settings"]["batch_size"])
NUM_WORKERS = int(config["settings"]["num_workers"])
NUM_EPOCHS = int(config["settings"]["num_epochs"])
REAL_NUM_EPOCHS = int(config["settings"]["real_num_epochs"])
LR_RATE = float(config["settings"]["lr_rate"])
MASK_CHECKPOINT = config["settings"]["mask_checkpoint"]
CHECKPOINT = config["settings"]["checkpoint"] 
REAL_CHECKPOINT = config["settings"]["real_checkpoint"]
CNN = config["settings"]["cnn"]
LSTM_SIZE = int(config["settings"]["lstm_size"])
LSTM_LAYERS = int(config["settings"]["lstm_layers"])
FRAME_NUM = int(config["settings"]["frame_num"])
TIME = int(config["settings"]["time"])
OUTPUT_SIZE = int(config["settings"]["output_size"])
DATA_ROOT = config["directories"]["data_root"]
VIDEO_SUBDIR = config["directories"]["video_subdir"]
PARA_SUBDIR = config["directories"]["para_subdir"]
SAVE_ROOT = config["directories"]["save_root"]
REAL_ROOT = config["directories"]["real_root"]
REAL_SAVE_ROOT = config["directories"]["real_save_root"]
ETA_MIN = float(config["settings"]["eta_min"])
DROP_RATE = float(config["settings"]["drop_rate"])
W_DECAY = float(config["settings"]["weight_decay"])

wandb.init(project="viscosity estimation testing", reinit=True, resume="never", config= config)

# Masking train video
# videoToMask = videoToMask(checkpoint = MASK_CHECKPOINT, data_root=DATA_ROOT, video_subdir=VIDEO_SUBDIR, save_root=SAVE_ROOT, frame_num=FRAME_NUM)
# videoToMask.mask_videos()
# Masking real world video
# videoToMask = videoToMask(checkpoint = MASK_CHECKPOINT, data_root=REAL_ROOT, video_subdir=VIDEO_SUBDIR, save_root=REAL_SAVE_ROOT, frame_num=FRAME_NUM)
# videoToMask.mask_videos()

# train/val dataset split
video_paths = sorted(glob.glob(osp.join(DATA_ROOT, VIDEO_SUBDIR, "*.mp4"))) # change to SAVE_ROOT for masked training
para_paths = sorted(glob.glob(osp.join(DATA_ROOT, PARA_SUBDIR, "*.json")))

train_video_paths, val_video_paths = train_test_split(video_paths, test_size=0.2, random_state=37)
train_para_paths, val_para_paths = train_test_split(para_paths, test_size=0.2, random_state=37)

train_ds = VideoDataset(train_video_paths, train_para_paths, FRAME_NUM, TIME)
val_ds = VideoDataset(val_video_paths, val_para_paths, FRAME_NUM, TIME)

# Load data
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)

# Initialize the optimizer and loss function
visc_model = ViscosityEstimator(LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE, DROP_RATE)
# visc_model = ViscosityResnet(OUTPUT_SIZE), only used for resnet based training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
visc_model.to(device)

optimizer = torch.optim.Adam(visc_model.parameters(), lr=LR_RATE, weight_decay=W_DECAY)

# criterion = nn.MSELoss()
criterion = MSLELoss(visc_model)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN)

wandb.watch(visc_model, criterion, log="all", log_freq=5)
# parameter Training Loop
num_epochs = NUM_EPOCHS
for epoch in range(num_epochs):  
    train_losses = []
    
    print(f"Epoch {epoch+1}/{num_epochs} - Training ")  
    visc_model.train()
    for frames, parameters in tqdm(train_dl):
        frames, parameters = frames.to(device), parameters.to(device) # (B, F, C, H, W)  (B, P)
        frames.requires_grad = True
        parameters.requires_grad = True

        outputs = visc_model(frames)
        train_loss = criterion(outputs, parameters)
        train_losses.append(train_loss.item())

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # loss print
        if (len(train_losses)) % 10 == 0:
            mean_train_loss = mean(train_losses)
            wandb.log({"train_loss": mean_train_loss})
    train_losses.clear()
    
    # Validation loss calculation
    visc_model.eval()
    val_losses = []
    with torch.no_grad():
        print(f"Epoch {epoch+1}/{num_epochs} - Validation")
    
    for frames, parameters in tqdm(val_dl):
        frames, parameters = frames.to(device), parameters.to(device)
        outputs = visc_model(frames)

        val_loss = criterion(outputs, parameters)
        val_losses.append(val_loss.item())
    mean_val_loss = mean(val_losses)
    wandb.log({"val_loss": mean_val_loss})

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{num_epochs} results - Train Loss: {mean_train_loss:.4f} Validation Loss: {mean_val_loss:.4f} - LR: {current_lr:.5f}")
wandb.finish()

# Save the model
torch.save(visc_model.state_dict(), CHECKPOINT)

"""
# Real World data loader

# train/val dataset split
real_video_paths = sorted(glob.glob(osp.join(REAL_SAVE_ROOT, "*.mp4")))
real_train_paths, real_val_paths = train_test_split(video_paths, test_size=0.2, random_state=42)

real_train_ds = VideoDataset(real_train_paths, frame_limit=FRAME_NUM*TIME)
real_val_ds = VideoDataset(real_val_paths, frame_limit=FRAME_NUM*TIME)

# Load data
real_train_dl = DataLoader(real_train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
real_val_dl = DataLoader(real_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Freeze CNN and LSTM layers
visc_model.load_state_dict(torch.load(CHECKPOINT))

for param in visc_model.cnn.parameters():
    param.requires_grad = False
for param in visc_model.lstm.parameters():
    param.requires_grad = False

# Fine Tuning loop definition
visc_model.fc = nn.Linear(visc_model.fc.in_features, 1).to(device)
optimizer = torch.optim.Adam(visc_model.fc.parameters(), lr=1e-4) #train only fc layer
criterion = nn.MSELoss()

# Fine Tuning Loop
num_epochs = REAL_NUM_EPOCHS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

visc_model.to(device)
visc_model.train()
for epoch in range(num_epochs):  
    train_losses = []
    print(f"Epoch {epoch+1}/{num_epochs} - Training ")
    for frames, parameters in real_train_dl:
        frames, parameters = frames.to(device), parameters.to(device)
        
        outputs = visc_model(frames)

        #loss calculation/optimization
        train_loss = reg_loss(outputs, parameters)
        train_losses.append(train_loss.item())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # loss print
        if (len(train_losses)) % 50 == 0:
            mean_train_loss = mean(train_losses)
            wandb.log({"real_train_loss": mean_train_loss})
    train_losses.clear()

    # Validation loss calculation
    visc_model.eval()
    val_losses = []
    with torch.no_grad():
        print(f"Epoch {epoch+1}/{num_epochs} - Validation")

    for frames, parameters in real_val_dl:
        frames, parameters = frames.to(device), parameters.to(device)
        outputs = visc_model(frames)

        val_loss = reg_loss(outputs, parameters)
        val_losses.append(val_loss.item())
    mean_val_loss = mean(val_losses)
    wandb.log({"real_val_loss": mean_val_loss})

    scheduler.step()
    current_lr = optimizer.get_last_lr[0]
    print(f"Epoch {epoch+1}/{num_epochs} results - Train Loss: {mean_train_loss:.4f} Validation Loss: {mean_val_loss:.4f} - LR: {current_lr:.5f}")
wandb.finish() 


# Save the model
torch.save(visc_model.state_dict(), REAL_CHECKPOINT)

"""