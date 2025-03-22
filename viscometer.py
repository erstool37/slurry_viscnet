import cv2
import yaml
import torch
import numpy as np
import os.path as osp
import glob

from src.model.ViscosityEstimator import ViscosityEstimator
from src.utils.VideoDataset import VideoDataset
from torch.utils.data import TensorDataset, DataLoader
from src.utils.paraPreprocess import logdescaler, zdescaler
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

CHECKPOINT = "src/model/weights_reg/ViscSyn0321_02.pth" 
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
BATCH_SIZE = int(config["settings"]["batch_size"])
NUM_WORKERS = int(config["settings"]["num_workers"])

video_path = sorted(glob.glob(osp.join(DATA_ROOT, VIDEO_SUBDIR, "data_1000.mp4")))
para_path = sorted(glob.glob(osp.join(DATA_ROOT, PARA_SUBDIR, "config_1000.json")))

ds = VideoDataset(video_path, para_path, FRAME_NUM, TIME)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)

# model load
visc_model = ViscosityEstimator(CNN, LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
visc_model.load_state_dict(torch.load(CHECKPOINT))
visc_model.eval()
visc_model.cuda()

for frames, parameters in dl:
    frames, parameters = frames.to(device), parameters.to(device)
    outputs = visc_model(frames)

    unnorm_outputs = torch.stack([zdescaler(outputs[:, 0], 'density'), logdescaler(outputs[:, 1], 'dynamic_viscosity'), zdescaler(outputs[:, 2], 'surface_tension')], dim=1)  # Shape: (batch, 3)
    unnorm_parameters = torch.stack([zdescaler(parameters[:, 0], 'density'), logdescaler(parameters[:, 1], 'dynamic_viscosity'), zdescaler(parameters[:, 2], 'surface_tension')], dim=1)  # Shape: (batch, 3)

    errors = (unnorm_outputs - unnorm_parameters) / unnorm_parameters * 100

    print("pred outputs", unnorm_outputs)
    print("ground_truth", unnorm_parameters)
    print("errors", errors)