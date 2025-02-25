import torch
import wandb
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import cv2
from torch.utils.data import Dataset, DataLoader, Subset
from preprocess.PreprocessorReg import pointToMesh, meshToVideo, videoToMask
from sklearn.model_selection import train_test_split
import os.path as osp
import glob
from statistics import mean
import yaml

# Load Config
with open("config_reg.yaml", "r") as file:
    config = yaml.safe_load(file)

BATCH_SIZE = config["settings"]["batch_size"]
NUM_WORKERS = config["settings"]["num_workers"]
NUM_EPOCHS = config["settings"]["num_epochs"]
REAL_NUM_EPOCHS = config["settings"]["real_num_epochs"]
LR_RATE = config["settings"]["lr_rate"]
MASK_CHECKPOINT = config["settings"]["mask_checkpoint"]
CHECKPOINT = config["settings"]["checkpoint"] 
REAL_CHECKPOINT = config["settings"]["real_checkpoint"]
CNN = config["settings"]["cnn"]
LSTM_SIZE = config["settings"][{"lstm_size"}]
LSTM_LAYERS = config["settings"]["lstm_layers"]
FRAME_NUM = config["settings"]["frame_num"]
TIME = config["settings"]["time"]
OUTPUT_SIZE = config["settings"]["output_size"]
DATA_ROOT = config["directories"]["data_root"]
POINT_SUBDIR = config["directories"]["point_subdir"]
MESH_SUBDIR = config["directories"]["mesh_subdir"]
VIDEO_SUBDIR = config["directories"]["video_subdir"]
PARA_SUBDIR = config["directories"]["para_subdir"]
SAVE_ROOT = config["directories"]["save_root"]
REAL_ROOT = config["directories"]["real_root"]
REAL_SAVE_ROOT = config["directories"]["real_save_root"]

wandb.init(project="viscosity estimation", reinit=True, resume="never", config= config)

# Mesh Formation
pointToMesh = pointToMesh(data_root=DATA_ROOT, point_subdir=POINT_SUBDIR, mesh_subdir=MESH_SUBDIR)
pointToMesh.pointToMesh()
# Video Formation
meshToVideo = meshToVideo(data_root=DATA_ROOT, mesh_subdir=MESH_SUBDIR, video_subdir=VIDEO_SUBDIR)
meshToVideo.meshToVideo()
# Masking train video
videoToMask = videoToMask(checkpoint = MASK_CHECKPOINT, data_root=DATA_ROOT, video_subdir=VIDEO_SUBDIR, save_root=SAVE_ROOT, frame_num=FRAME_NUM)
videoToMask.mask_videos()
# Masking real world video
videoToMask = videoToMask(checkpoint = MASK_CHECKPOINT, data_root=REAL_ROOT, video_subdir=VIDEO_SUBDIR, save_root=REAL_SAVE_ROOT, frame_num=FRAME_NUM)
videoToMask.mask_videos()

class VideoDataset(Dataset):
    def __init__(self, video_paths, para_paths, frame_limit=FRAME_NUM*TIME):
        '''Initialize dataset'''
        self.video_paths = video_paths
        self.para_paths = para_paths
        self.frame_limit = frame_limit

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        video_path = self.video_paths[index]
        para_path = self.para_paths[index]
        
        frames = self.__loadvideo__(video_path, self.frame_limit)
        frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)  # (T, C, H, W)

        parameters = self.__loadparameters__(para_path)

        return frames, parameters
    
    def __loadvideo__(self, video_path, frame_limit=32):
        cap = cv2.VideoCapture(video_path)
        frames = []
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or len(frames) >= frame_limit:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
            frames.append(frame)
        cap.release()

        while len(frames) < frame_limit:
            frames.append(np.zeros_like(frames[0]))  # Add empty frames if needed
    
        return np.array(frames, dtype=np.uint8) # (T, C, H, W)
    
    def __loadparameters__(self, para_path):
        parameters = torch.tensor(np.load(para_path), dtype=torch.float32).squeeze(0)  # (1, N)

        return parameters

# train/val dataset split
video_paths = sorted(glob.glob(osp.join(SAVE_ROOT, "*.mp4")))
para_paths = sorted(glob.glob(osp.join(DATA_ROOT, PARA_SUBDIR, "*npy")))

train_ds = VideoDataset(video_paths, para_paths, frame_limit=FRAME_NUM*TIME)
val_ds = VideoDataset(para_paths, para_paths, frame_limit=FRAME_NUM*TIME)
print(train_ds[0][1].shape)

indices = np.arange(len(train_ds))
train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=True, random_state=42)

train_ds = Subset(train_ds, train_idx)
val_ds = Subset(val_ds, val_idx)

# Load data
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Model Defining
class ViscosityEstimator(nn.Module):
    def __init__(self, cnn_model=CNN, lstm_hidden_size=LSTM_SIZE, lstm_layers=LSTM_LAYERS, output_size=OUTPUT_SIZE):
        super(ViscosityEstimator, self).__init__()
        # CNN
        self.cnn = getattr(models, cnn_model)(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.cnn_out_features = 512
        
        # LSTM
        self.lstm = nn.LSTM(input_size=self.cnn_out_features, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
        
        # Viscosity prediction
        self.fc = nn.Linear(lstm_hidden_size, output_size)
    
    def forward(self, x):
        """ x: (batch_size, sequence_length, channels, height, width)"""
        batch_size, seq_len, C, H, W = x.size()

        cnn_features = []
        for t in range(seq_len): 
            frame_features = self.cnn(x[:, t, :, :, :])  # if too slow, batch flattening=
            cnn_features.append(frame_features.view(batch_size, -1))
        cnn_features = torch.stack(cnn_features, dim=1) 

        # LSTM Processing
        lstm_out, _ = self.lstm(cnn_features)  # Output: (batch_size, seq_len, hidden_size)
        lstm_last_out = lstm_out[:, -1, :]  # Take last time step output
        
        viscosity = self.fc(lstm_last_out)  # (batch_size, 1)
        
        return viscosity

# Initialize the optimizer and loss function
visc_model = ViscosityEstimator()

optimizer = torch.optim.Adam(visc_model.parameters(), lr=LR_RATE, weight_decay=0)
reg_loss = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# parameter Training Loop
num_epochs = NUM_EPOCHS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
visc_model.to(device)
visc_model.train()

for epoch in range(num_epochs):  
    train_losses = []
    print(f"Epoch {epoch+1}/{num_epochs} - Training ")
    for frames, parameters in train_dl:
        frames, parameters = frames.to(device), parameters.to(device)
        
        outputs = visc_model(frames)
        print(outputs.shape)

        #loss calculation/optimization
        train_loss = reg_loss(outputs, parameters)
        train_losses.append(train_loss.item())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # loss print
        if (len(train_losses)) % 50 == 0:
            mean_train_loss = mean(train_losses)
            wandb.log({"train_loss": mean_train_loss})
    train_losses.clear()

    # Validation loss calculation
    visc_model.eval()
    val_losses = []
    with torch.no_grad():
        print(f"Epoch {epoch+1}/{num_epochs} - Validation")

    for frames, parameters in val_dl:
        frames, parameters = frames.to(device), parameters.to(device)
        outputs = visc_model(frames)

        val_loss = reg_loss(outputs, parameters)
        val_losses.append(val_loss.item())
    mean_val_loss = mean(val_losses)
    wandb.log({"val_loss": mean_val_loss})

    scheduler.step()
    current_lr = optimizer.get_last_lr[0]
    print(f"Epoch {epoch+1}/{num_epochs} results - Train Loss: {mean_train_loss:.4f} Validation Loss: {mean_val_loss:.4f} - LR: {current_lr:.5f}")
wandb.finish() 

# Save the model
torch.save(visc_model.state_dict(), CHECKPOINT)

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