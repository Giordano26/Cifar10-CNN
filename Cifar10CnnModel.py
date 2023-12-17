# Define the CNN architecture
import torch.nn as nn
from ImageClassificationBase import ImageClassificationBase

class Cifar10CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 64 x 16 x 16 / 64 channles with 16x16 px
            nn.Dropout(0.25),  # Added Dropout for preventing overfitting

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 128 x 8 x 8 / 128 channles with 8x8 px
            nn.Dropout(0.25),  # Added Dropout for preventing overfitting

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 256 x 4 x 4 / 256 channles with 4x4 px
            nn.Dropout(0.25),  # Added Dropout for preventing overfitting

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),  # More Dropout for regularization before the linear layer
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.25),  # More Dropout for regularization before the linear layer
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)