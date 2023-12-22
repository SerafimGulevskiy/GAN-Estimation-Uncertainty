import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import torchvision

class Base_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, info):
        x = torch.cat([x, info], 1)
        output = self.model(x)
        return output
    
    
class Base_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),#2 = 1(space_dim/noise_dim) + 1(additional info, x)
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, info):
        x = torch.cat([x, info], 1)
        output = self.model(x)
        return output