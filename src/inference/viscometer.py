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
import sys
import torch.optim as optim
from tqdm import tqdm
from statistics import mean
import importlib
import yaml
import json
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from utils import MAPEcalculator, MAPEflowcalculator, MAPEtestcalculator, set_seed, distribution
from dataset import VideoDataset

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, default="configs/testconfig.yaml")
args = parser.parse_args()

with open("configs/testconfig.yaml", "r") as file:
    config = yaml.safe_load(file)
cfg = config["regression"]

SCALER          = cfg["preprocess"]["scaler"]
DESCALER        = cfg["preprocess"]["descaler"]
TEST_SIZE       = float(cfg["preprocess"]["test_size"])
RAND_STATE      = int(cfg["preprocess"]["random_state"])
FRAME_NUM       = int(cfg["preprocess"]["frame_num"])
TIME            = int(cfg["preprocess"]["time"])
BATCH_SIZE      = int(cfg["train_settings"]["batch_size"])
NUM_WORKERS     = int(cfg["train_settings"]["num_workers"])
NUM_EPOCHS      = int(cfg["train_settings"]["num_epochs"])
SEED            = int(cfg["train_settings"]["seed"])
ENCODER         = cfg["model"]["encoder"]["encoder"]
CNN             = cfg["model"]["encoder"]["cnn"]
CNN_TRAIN       = cfg["model"]["encoder"]["cnn_train"]
LSTM_SIZE       = int(cfg["model"]["encoder"]["lstm_size"])
LSTM_LAYERS     = int(cfg["model"]["encoder"]["lstm_layers"])
OUTPUT_SIZE     = int(cfg["model"]["encoder"]["output_size"])
DROP_RATE       = float(cfg["model"]["encoder"]["drop_rate"])
FLOW            = cfg["model"]["flow"]["flow"]
FLOW_BOOL       = cfg["model"]["flow"]["flow_bool"]
DIM             = int(cfg["model"]["flow"]["dim"])
CON_DIM         = int(cfg["model"]["flow"]["con_dim"])
HIDDEN_DIM      = int(cfg["model"]["flow"]["hidden_dim"])
NUM_LAYERS      = int(cfg["model"]["flow"]["num_layers"])
CHECKPOINT      = cfg["directories"]["checkpoint"]["checkpoint"]
DATA_ROOT       = cfg["directories"]["data"]["data_root"]
VIDEO_SUBDIR    = cfg["directories"]["data"]["video_subdir"]
PARA_SUBDIR     = cfg["directories"]["data"]["para_subdir"]
NORM_SUBDIR     = cfg["directories"]["data"]["norm_subdir"]
SAVE_ROOT       = cfg["directories"]["data"]["save_root"]
TEST_ROOT       = cfg["directories"]["data"]["test_root"]

video_paths = sorted(glob.glob(osp.join(DATA_ROOT, VIDEO_SUBDIR, "*.mp4")))
para_paths = sorted(glob.glob(osp.join(DATA_ROOT, NORM_SUBDIR, "*.json")))
# test_video_paths = sorted(glob.glob(osp.join(TEST_ROOT, VIDEO_SUBDIR, "*.mp4")))
# test_para_paths = sorted(glob.glob(osp.join(TEST_ROOT, PARA_SUBDIR, "*.json")))

train_video_paths, val_video_paths = train_test_split(video_paths, test_size=TEST_SIZE, random_state=RAND_STATE)
train_para_paths, val_para_paths = train_test_split(para_paths, test_size=TEST_SIZE, random_state=RAND_STATE)

train_ds = VideoDataset(train_video_paths, train_para_paths, FRAME_NUM, TIME)
val_ds = VideoDataset(val_video_paths, val_para_paths, FRAME_NUM, TIME)
# test_ds = VideoDataset(test_video_paths, test_para_paths, FRAME_NUM, TIME)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)
# test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)

# model load
encoder_module = importlib.import_module(f"models.{ENCODER}")
flow_module = importlib.import_module(f"models.{FLOW}")

encoder_class = getattr(encoder_module, ENCODER)
flow_class = getattr(flow_module, FLOW)

encoder = encoder_class(LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE, DROP_RATE, CNN, CNN_TRAIN, FLOW_BOOL)
flow = flow_class(DIM, CON_DIM, HIDDEN_DIM, NUM_LAYERS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.cuda()
encoder.eval()
encoder.load_state_dict(torch.load(CHECKPOINT))

# Error Calculation
errors = []
for frames, parameters in val_dl:
    frames, parameters = frames.to(device), parameters.to(device)
    outputs = encoder(frames)

    if FLOW_BOOL:
        z, log_det_jacobian = flow(parameters, outputs)
        visc = flow.inverse(z, outputs)
        error = MAPEtestcalculator(visc.detach(), parameters.detach(), DESCALER, "val", DATA_ROOT)
    else:
        error = MAPEtestcalculator(outputs.detach(), parameters.detach(), DESCALER, "val", DATA_ROOT)
    errors.append(error.detach().cpu())

errors_tensor = torch.cat(errors, dim=0)
meanerror = errors_tensor.mean(dim=0)  # shape: [3]

distribution(data=errors_tensor[:,0], ref = 0, save_path='src/inference/error/dist_den.png')
distribution(data=errors_tensor[:,1], ref = 0, save_path='src/inference/error/dist_visco.png')
distribution(data=errors_tensor[:,2], ref = 0, save_path='src/inference/error/dist_surf.png')

print(f"density MAPE: {float(meanerror[0]):.2f}%")
print(f"dynamic viscosity MAPE: {float(meanerror[1]):.2f}%")
print(f"surface tension MAPE: {float(meanerror[2]):.2f}%")

# Regression Validation test
"""
unnorm_outputs_list = []
unnorm_para_list = []

for frames, parameters in test_dl:
    frames, parameters = frames.to(device), parameters.to(device)
    outputs = visc_model(frames)
    
    unnorm_outputs = torch.stack([zdescaler(outputs[:, 0], 'density'), zdescaler(outputs[:, 1], 'dynamic_viscosity'), zdescaler(outputs[:, 2], 'surface_tension')], dim=1)  
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
    distribution(grouped_outputs_list[idx][:,0], ref = grouped_para_list[idx][0,0].cpu(), save_path=f'test/precision/dist{(idx+1):02d}_den.png')
    distribution(grouped_outputs_list[idx][:,1], ref = grouped_para_list[idx][0,1].cpu(), save_path=f'test/precision/dist{(idx+1):02d}_visco.png')
    distribution(grouped_outputs_list[idx][:,2], ref = grouped_para_list[idx][0,2].cpu(), save_path=f'test/precision/dist{(idx+1):02d}_surf.png')
"""