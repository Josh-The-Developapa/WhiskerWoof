import torch
from torch import nn


class WhiskerWoof(nn.Module):
    """Convolutional Neural Network for classifying 200x200 images of Cats and Dogs"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=5)
        self.linear_stack = nn.Sequential(
            nn.Linear(44180, 240),  # 44180 = 47*47*20
            nn.ReLU(),
            nn.Linear(240, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))  # -> Outputs 16 feature maps of size 196x196
        x = self.pool(x)  # -> Outputs 16 downsized feature maps of size 98x98
        x = self.relu(self.conv2(x))  # -> Outputs 20 Feature maps of size 94x94
        x = self.pool(x)  # -> Outputs 20 downsized feature maps of size 47x47
        x = nn.Flatten()(x)  # -> 47*47*20 = 44180 neurons
        x = self.linear_stack(x)
        return x
