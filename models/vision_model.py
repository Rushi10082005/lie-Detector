# models/vision_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionCNN(nn.Module):
    """
    A simple Convolutional Neural Network for vision-based analysis.
    It takes an image and outputs a score for 'lie' vs 'truth'.
    """
    def __init__(self):
        super(VisionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Assuming input images are 64x64, after 2 pools -> 16x16
        # The linear layer input size is 32 channels * 16 * 16 pixels
        self.fc1 = nn.Linear(32 * 16 * 16, 2)  # 2 outputs: [lie_score, truth_score]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x