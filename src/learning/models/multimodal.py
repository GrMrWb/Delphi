"""
This is an exmaple of the multi modal model
"""

import torch
import torch.nn as nn

class MultiModalModel(nn.Module):
    def __init__(self, text_input_dim, image_input_dim, audio_input_dim, output_dim):
        super(MultiModalModel, self).__init__()
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 64)  # Adjust based on your image size
        )
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * 250, 64)  # Adjust based on your audio input size
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, data):
        text_features = self.text_encoder(data["text"])
        image_features = self.image_encoder(data["image"])
        audio_features = self.audio_encoder(data["audio"])
        
        # Concatenate features
        combined = torch.cat((text_features, image_features, audio_features), dim=1)
        
        # Fusion
        output = self.fusion(combined)
        
        return output