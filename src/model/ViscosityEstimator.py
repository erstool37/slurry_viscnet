import torch.nn as nn
import torch
from torchvision import models

class ViscosityEstimator(nn.Module):
    def __init__(self, cnn_model, lstm_hidden_size, lstm_layers, output_size):
        super(ViscosityEstimator, self).__init__()
        self.cnn = getattr(models, cnn_model)(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.cnn_out_features = 512
        
        # LSTM
        self.lstm = nn.LSTM(input_size=self.cnn_out_features, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
        
        # Viscosity prediction
        self.fc = nn.Linear(lstm_hidden_size, output_size)
    
    def forward(self, x):
        """ x: (batch_size, sequence_length, channels, height, width)"""
        batch_size, frames, C, H, W = x.size()
        x = x.view(batch_size * frames, C, H, W)
        video_features = self.cnn(x) # VRAM 27MB
        video_features = video_features.view(batch_size, frames, -1) 

        lstm_out, _ = self.lstm(video_features)
        lstm_last_out = lstm_out[:, -1, :]  # Take last time step output
        
        viscosity = self.fc(lstm_last_out)  # (batch_size, output_size)
        
        return viscosity