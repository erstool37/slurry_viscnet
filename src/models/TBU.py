



# PySlowFast? for 3DResnet


# Attention based models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = input_dim ** 0.5

    def forward(self, x):
        # x: (batch_size, frames, input_dim)
        Q = self.query(x)  # (batch_size, frames, input_dim)
        K = self.key(x)    # (batch_size, frames, input_dim)
        V = self.value(x)  # (batch_size, frames, input_dim)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, frames, frames)
        attention_weights = F.softmax(attention_scores, dim=-1)  # Normalize scores
        
        # Apply attention weights
        context = torch.matmul(attention_weights, V)  # (batch_size, frames, input_dim)
        context = context.mean(dim=1)  # Aggregate across frames -> (batch_size, input_dim)

        return context

class ViscosityResnet(nn.Module):
    def __init__(self, output_size):
        super(ViscosityResnet, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.cnn = nn.Sequential(*list(self.resnet.children())[:-1])
        self.cnn_out_features = 512
        
        # Attention mechanism over frame-level features
        self.attention = Attention(self.cnn_out_features)
        self.fc = nn.Linear(self.cnn_out_features, output_size)

    def forward(self, x):
        batch_size, frames, C, H, W = x.size()
        x = x.view(batch_size * frames, C, H, W)

        # Extract frame-level features
        video_features = self.cnn(x)  # (batch_size * frames, 512, 1, 1)
        video_features = video_features.view(batch_size, frames, -1)  # (batch_size, frames, 512)

        # Apply attention over frames
        context = self.attention(video_features)  # (batch_size, 512)

        # Final prediction
        viscosity = self.fc(context)  # (batch_size, output_size)
        return viscosity