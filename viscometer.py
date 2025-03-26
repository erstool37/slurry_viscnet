import cv2
import yaml
import torch
import numpy as np
import os.path as osp
import glob
from sklearn.model_selection import train_test_split
from src.model.ViscosityEstimator import ViscosityEstimator
from src.model.ViscosityResnet import ViscosityResnet
from src.utils.VideoDataset import VideoDataset
from torch.utils.data import TensorDataset, DataLoader
from src.utils.PreprocessorPara import logdescaler, zdescaler
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns
# from preprocess.mobile_sam import sam_model_registry, SamPredictor

# 1. Inference for WebGL Segmentation
"""
vortex_model = sam_model_registry["vit_t"](checkpoint="preprocess/model_seg/Vortex0219_01.pth")
vortex_model.eval()
vortex_model.cuda()

#image setting
test_image = cv2.imread("test_seg.jpg")
test_image = test_image[0:1024, 0:1024] # required because predictor.py does not use postprocess yet

#predict
predictor = SamPredictor(vortex_model)
predictor.set_image(image=test_image, image_format="RGB") # normalize RGB, padding, make tensor
mask, _, _ = predictor.predict(multimask_output = False, return_logits=True) # get (256, 256) masked image with 0~255 uint8 format

cv2.imwrite("test_mask.jpg", mask)
"""

# 2. Inference for CFD Viscosity Estimation

with open("config_reg.yaml", "r") as file:
    config = yaml.safe_load(file)

CHECKPOINT = "src/model/weights_reg/ViscSyn0326_03.pth" 
LSTM_SIZE = int(config["settings"]["lstm_size"])
LSTM_LAYERS = int(config["settings"]["lstm_layers"])
FRAME_NUM = int(config["settings"]["frame_num"])
TIME = int(config["settings"]["time"])
OUTPUT_SIZE = int(config["settings"]["output_size"])
DATA_ROOT = config["directories"]["data_root"]
VIDEO_SUBDIR = config["directories"]["video_subdir"]
PARA_SUBDIR = config["directories"]["para_subdir"]
BATCH_SIZE = int(config["settings"]["batch_size"])
NUM_WORKERS = int(config["settings"]["num_workers"])
# TEST_ROOT = 

video_paths = sorted(glob.glob(osp.join(DATA_ROOT, VIDEO_SUBDIR, "*.mp4")))
para_paths = sorted(glob.glob(osp.join(DATA_ROOT, PARA_SUBDIR, "*.json")))
# test_video_paths = sorted(glob.glob(osp.join(TEST_ROOT, VIDEO_SUBDIR, "mp4")))
# test_para_paths = sorted(glob.glob(osp.join(TEST_ROOT, VIDEO_SUBDIR, "json")))

train_video_paths, val_video_paths = train_test_split(video_paths, test_size=0.2, random_state=37)
train_para_paths, val_para_paths = train_test_split(para_paths, test_size=0.2, random_state=37)

train_ds = VideoDataset(train_video_paths, train_para_paths, FRAME_NUM, TIME)
val_ds = VideoDataset(val_video_paths, val_para_paths, FRAME_NUM, TIME)
# test_ds = VideoDataset(test_video_paths, test_para_paths, FRAME_NUM, TIME)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)
# test_dl = DataLoader(test_ds, batch_size=40, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)

# model load
visc_model = ViscosityEstimator(LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE)
# visc_model = ViscosityResnet(OUTPUT_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
visc_model.load_state_dict(torch.load(CHECKPOINT))
visc_model.eval()
visc_model.cuda()

errors = []

# Error Calculation
for frames, parameters in val_dl:
    frames, parameters = frames.to(device), parameters.to(device)
    outputs = visc_model(frames)

    # zscale, log scale mixed version
    # unnorm_outputs = torch.stack([zdescaler(outputs[:, 0], 'density'), logdescaler(outputs[:, 1], 'dynamic_viscosity'), zdescaler(outputs[:, 2], 'surface_tension')], dim=1)  # Shape: (batch, 3)
    # unnorm_parameters = torch.stack([zdescaler(parameters[:, 0], 'density'), logdescaler(parameters[:, 1], 'dynamic_viscosity'), zdescaler(parameters[:, 2], 'surface_tension')], dim=1)  # Shape: (batch, 3)
    
    # log scaling version
    unnorm_outputs = torch.stack([logdescaler(outputs[:, 0], 'density'), logdescaler(outputs[:, 1], 'dynamic_viscosity'), logdescaler(outputs[:, 2], 'surface_tension')], dim=1)  
    unnorm_parameters = torch.stack([logdescaler(parameters[:, 0], 'density'), logdescaler(parameters[:, 1], 'dynamic_viscosity'), logdescaler(parameters[:, 2], 'surface_tension')], dim=1)

    error = (unnorm_outputs - unnorm_parameters) / abs(unnorm_parameters) * 100
    errors.append(error)

errors = torch.cat(errors, dim=0) # error concat
meanerror = torch.mean(torch.abs(errors), dim=0) # mean, absolute error calculation

# histogram of errors
def distribution(data, title='Normalized Value Distribution', save_path='test.png'):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    plt.figure()
    sns.histplot(data, kde=True, bins=50, stat='density', edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(save_path)
    plt.close()

distribution(errors[:,0], save_path='test/errordist_den.png')
distribution(errors[:,1], save_path='test/errordist_visco.png')
distribution(errors[:,2], save_path='test/errordist_surf.png')

# MAPE printing
print(f"density MAPE: {float(meanerror[0]):.2f}%")
print(f"dynamic viscosity MAPE: {float(meanerror[1]):.2f}%")
print(f"surface tension MAPE: {float(meanerror[2]):.2f}%")

"""
# Regression Validation test
unnorm_outputs_list = []
for frames, parameters in test_dl:
    frames, parameters = frames.to(device), parameters.to(device)
    outputs = visc_model(frames)
    
    unnorm_outputs = torch.stack([logdescaler(outputs[:, 0], 'density'), logdescaler(outputs[:, 1], 'dynamic_viscosity'), logdescaler(outputs[:, 2], 'surface_tension')], dim=1)  
    unnorm_parameters = torch.stack([logdescaler(parameters[:, 0], 'density'), logdescaler(parameters[:, 1], 'dynamic_viscosity'), logdescaler(parameters[:, 2], 'surface_tension')], dim=1)
    unnorm_outputs_list.append(unnorm_outputs) 

    # error = (unnorm_outputs - unnorm_parameters) / abs(unnorm_parameters) * 100
    # errors.append(error)

for idx in len(unnorm_outputs_list):
    distribution(unnorm_outputs_list[idx][0], save_path=f'test/testdist{idx+1}_den.png')
    distribution(unnorm_outputs_list[idx][2], save_path=f'test/testdist{idx+1}_visco.png')
    distribution(unnorm_outputs_list[idx][3], save_path=f'test/testdist{idx+1}_surf.png')
"""