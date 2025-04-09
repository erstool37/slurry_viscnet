import torch
import torch.nn as nn
from torchvision import models
import datetime
import cv2
import wandb
import argparse
import numpy as np
import os.path as osp
import glob
import torch.optim as optim
from tqdm import tqdm
from statistics import mean
import importlib
import yaml
import json
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from datasets.VideoDataset import VideoDataset
from utils.utils import MAPEcalculator
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)
cfg = config["regression"]

SCALER          = cfg["preprocess"]["scaler"]
DESCALER        = cfg["preprocess"]["descaler"]
BATCH_SIZE      = int(cfg["train_settings"]["batch_size"])
NUM_WORKERS     = int(cfg["train_settings"]["num_workers"])
CHECKPOINT      = "src/weights/5s_10fps_visconly_100xvisc_run_init_0403_v1.pth"
DESCALER        = cfg["preprocess"]["descaler"]
MODEL           = cfg["model"]["model_class"]
CNN             = cfg["model"]["cnn"]
CNN_TRAIN       = cfg["model"]["cnn_train"]
LSTM_SIZE       = int(cfg["model"]["lstm_size"])
LSTM_LAYERS     = int(cfg["model"]["lstm_layers"])
FRAME_NUM       = int(cfg["preprocess"]["frame_num"])
# TIME            = int(cfg["preprocess"]["time"])
TIME            = 3
TEST_SIZE       = float(cfg["preprocess"]["test_size"])
RAND_STATE      = int(cfg["preprocess"]["random_state"])
OUTPUT_SIZE     = int(cfg["model"]["output_size"])
DROP_RATE       = float(cfg["model"]["drop_rate"])
DATA_ROOT       = cfg["directories"]["data"]["data_root"]
VIDEO_SUBDIR    = cfg["directories"]["data"]["video_subdir"]
PARA_SUBDIR     = cfg["directories"]["data"]["para_subdir"]
NORM_SUBDIR     = cfg["directories"]["data"]["norm_subdir"]
TEST_ROOT       = cfg["directories"]["data"]["test_root"]
SAVE_ROOT       = cfg["directories"]["data"]["save_root"]
REAL_ROOT       = cfg["directories"]["data"]["real_root"]
REAL_SAVE_ROOT  = cfg["directories"]["data"]["real_save_root"]
TEST_CHECKPOINT = "src/weights/5s_10fps_visconly_100xvisc_run_init_0403_v1.pth"

video_paths = sorted(glob.glob(osp.join(DATA_ROOT, VIDEO_SUBDIR, "*.mp4")))
para_paths = sorted(glob.glob(osp.join(DATA_ROOT, NORM_SUBDIR, "*.json")))
test_video_paths = sorted(glob.glob(osp.join(TEST_ROOT, VIDEO_SUBDIR, "*.mp4")))
test_para_paths = sorted(glob.glob(osp.join(TEST_ROOT, PARA_SUBDIR, "*.json")))

train_video_paths, val_video_paths = train_test_split(video_paths, test_size=0.05, random_state=RAND_STATE)
train_para_paths, val_para_paths = train_test_split(para_paths, test_size=0.05, random_state=RAND_STATE)

train_ds = VideoDataset(train_video_paths, train_para_paths, FRAME_NUM, TIME)
val_ds = VideoDataset(val_video_paths, val_para_paths, FRAME_NUM, TIME)
test_ds = VideoDataset(test_video_paths, test_para_paths, FRAME_NUM, TIME)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)

# model load
model_module = importlib.import_module(f"models.{MODEL}")
model_class = getattr(model_module, MODEL)
visc_model = model_class(LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE, DROP_RATE, CNN, CNN_TRAIN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
visc_model.cuda()
visc_model.eval()
visc_model.load_state_dict(torch.load(TEST_CHECKPOINT))

# histogram of errors
def distribution(data, ref=None, title='Normalized Value Distribution', save_path='.', prefix='dist'):
    import os
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    plt.figure(figsize=(8, 3))
    ax = sns.histplot(data, kde=True, bins=50, stat='density', edgecolor='black')
    if ref is not None:
        ymax = ax.get_ylim()[1]
        plt.vlines(ref, ymin=0, ymax=ymax, color='red', linestyle='--', label=f'Ref: {ref}')
        plt.legend()
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('probability Density')
    plt.savefig(save_path)
    plt.close()

# Regression TEST
unnorm_outputs_list = []
unnorm_para_list = []

for frames, parameters in test_dl:
    frames, parameters = frames.to(device), parameters.to(device)
    outputs = visc_model(frames)

    utils = importlib.import_module("utils")
    descaler = getattr(utils, DESCALER)

    unnorm_outputs = torch.stack([descaler(outputs[:, 0], 'density', TEST_ROOT), descaler(outputs[:, 1], 'dynamic_viscosity', TEST_ROOT), descaler(outputs[:, 2], 'surface_tension', TEST_ROOT)], dim=1)  
    unnorm_para = torch.stack([parameters[:, 0], parameters[:, 1], parameters[:, 2]], dim=1)
    
    unnorm_outputs_list.append(unnorm_outputs.detach().cpu()) 
    unnorm_para_list.append(unnorm_para.detach().cpu())

unnorm_outputs_list = torch.cat(unnorm_outputs_list, dim=0)
unnorm_para_list = torch.cat(unnorm_para_list, dim=0)

groups = defaultdict(list)

for idx, item in enumerate(unnorm_para_list):
    key = item[0].item()
    groups[key].append(idx)
grouped_indices = list(groups.values())

grouped_outputs_list = [unnorm_outputs_list[idx] for idx in grouped_indices]
grouped_para_list = [unnorm_para_list[idx] for idx in grouped_indices]

for idx in range(len(grouped_outputs_list)):
    distribution(grouped_outputs_list[idx][:,0], ref = grouped_para_list[idx][0,0].cpu(), save_path=f'{(idx+1):02d}_den.png')
    distribution(grouped_outputs_list[idx][:,1], ref = grouped_para_list[idx][0,1].cpu(), save_path=f'{(idx+1):02d}_visco.png')
    distribution(grouped_outputs_list[idx][:,2], ref = grouped_para_list[idx][0,2].cpu(), save_path=f'{(idx+1):02d}_surf.png')