import torch.nn as nn
import torch
from torchvision import models

class ViscosityEstimator(nn.Module):
    def __init__(self, cnn_model, lstm_hidden_size, lstm_layers, output_size):
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