import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class BayesianViscosityEstimator(nn.Module):
    def __init__(self, lstm_hidden_size, lstm_layers, output_size, dropout):
        super(BayesianViscosityEstimator, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.resnet.children())[:-1])
        self.cnn_out_features = 512

        for param in self.cnn.parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(input_size=self.cnn_out_features, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True, dropout=dropout)
        
        self.fc_mu = nn.Linear(lstm_hidden_size, output_size)
        self.fc_sigma = nn.Linear(lstm_hidden_size, output_size)
    
    def forward(self, x):
        batch_size, frames, C, H, W = x.size()
        x = x.view(batch_size * frames, C, H, W)

        video_features = self.cnn(x)  # (B*T, 512, 1, 1)
        video_features = video_features.view(batch_size, frames, -1)  # (B, T, 512)

        lstm_out, _ = self.lstm(video_features)
        lstm_last_out = lstm_out[:, -1, :]  # (B, H)

        mu = self.fc_mu(lstm_last_out)                      # (B, output_size)
        sigma = F.softplus(self.fc_sigma(lstm_last_out)) + 1e-6

        return mu, sigma