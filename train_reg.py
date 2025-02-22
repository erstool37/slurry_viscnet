import torch
import wandb
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as data_utils
import torch.nn as nn
from torchvision import models
from preprocess.PreprocessorReg import pointToMesh, meshToVideo, videoToMask

# Model Definition
BATCH_SIZE = 16
NUM_WORKERS = 0
NUM_EPOCHS = 40
LR_RATE = 5e-4
MASK_CHECKPOINT = "preprocess/model_seg/Vortex0219_01.pth" # masking model name
CHECKPOINT = "preprocess/model_reg/Vortex0222_01.pth" # freshly trained model name
BASE_CHECKPOINT = "preprocess/model_reg/Vortex0222_01.pth" # pre-trained model name

CONFIG= {}

CONFIG["batch_size"] = BATCH_SIZE
CONFIG["learning_rate"] = LR_RATE
CONFIG["epochs"] = NUM_EPOCHS
CONFIG["architecture"] = "CNN-LSTM"
CONFIG["dataset"] = "CFDfluid synthetic data"
CONFIG["scheduler"] = "CosineAnnealingLR"
CONFIG["loss"] = "DiceBCELoss"
CONFIG["checkpoint"] = CHECKPOINT

# wandb.init(project="viscosity estimation", reinit=True, resume="never", config= CONFIG)

# Repository path
DATA_ROOT = "dataset/CFDfluid/original"
POINT_SUBDIR = "pointcloud"
MESH_SUBDIR = "mesh"
VIDEO_SUBDIR = "videos"
SAVE_ROOT = "dataset/CFDfluid/processed_data"
MASK_SAVE_SUBDIR = "masks"

# Mesh Formation
# pointToMesh = pointToMesh(data_root=DATA_ROOT, point_subdir=POINT_SUBDIR, mesh_subdir=MESH_SUBDIR)
# pointToMesh.pointToMesh()
# Video Formation
# meshToVideo = meshToVideo(data_root=DATA_ROOT, mesh_subdir=MESH_SUBDIR, video_subdir=VIDEO_SUBDIR)
# meshToVideo.meshToVideo()
# Masking

FRAME_NUM = 10 # desired number of masked frame per second 
print("hi")
videoToMask = videoToMask(checkpoint = MASK_CHECKPOINT, data_root=DATA_ROOT, video_subdir=VIDEO_SUBDIR, save_root=SAVE_ROOT, mask_save_subdir=MASK_SAVE_SUBDIR, frame_num=FRAME_NUM)
videoToMask.mask_videos()

# Load images and pile in dataset

# Split dataset into train and validation

# Initialize the optimizer
train_dataset = TensorDataset(video_tensor, labels_tensor)
train_loader = DataLoader(dataset = train_dataset, batch_size= 4, shuffle= True)

# Training loop
class ViscosityEstimator(nn.Module):
  
    def __init__(self, cnn_model='resnet18', lstm_hidden_size=128, lstm_layers=2):
        super(ViscosityEstimator, self).__init__()
        
        # Pre-trained CNN
        self.cnn = getattr(models, cnn_model)(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # 1st component is for classification > remove
        self.cnn_out_features = 512
        
        # LSTM
        self.lstm = nn.LSTM(input_size=self.cnn_out_features, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_layers, 
                            batch_first=True)
        
        # Fully connected
        self.fc = nn.Linear(lstm_hidden_size, 1)
    
    def forward(self, x):
        """ x: (batch_size, sequence_length, channels, height, width)"""
        batch_size, seq_len, channels, height, width = x.size()

        cnn_features = []

        for t in range(seq_len):
            frame = x[:, t, :, :, :] 
            frame_features = self.cnn(frame)
            frame_features = frame_features.view(batch_size, -1) 
            cnn_features.append(frame_features)
        
        cnn_features = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(cnn_features)
        lstm_last_out = lstm_out[:, -1, :]
        
        viscosity = self.fc(lstm_last_out)  # (batch_size, 1)
        
        return viscosity


import torch.optim as optim

model = ViscosityEstimator(cnn_model='resnet18', lstm_hidden_size=128, lstm_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


# Save the model

# Real World data loader

# Fine Tuning

# Save the model

