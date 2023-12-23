import torch
import math
from torch.utils.data import Dataset

class SinDataset(Dataset):
    def __init__(self, train_data_length=1024, train_data_length_certain=800):
        """
        MyDataset class for handling training data.

        Parameters:
        - train_data_length (int): Total length of the training data.
        - train_data_length_certain (int): Length of the certain part of the training data.
        """
        self.train_data_length = train_data_length
        self.train_data_length_certain = train_data_length_certain
        self.train_data_length_uncertain = train_data_length - train_data_length_certain

        # Initialize the training data tensor directly as an attribute
        self.data = torch.zeros((train_data_length, 1))

        self.data[:train_data_length_certain, 0] = 1.5 * math.pi * torch.rand(train_data_length_certain)

        self.data[train_data_length_certain:, 0] = 2 * math.pi * torch.rand(self.train_data_length_uncertain)

        # Store the sine values of the entire training data
        self.data = [self.data]
        self.data.insert(0, torch.sin(self.data[0][:, 0]))


    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        x = self.data[0][idx]
        y = self.data[1][idx]
        return x, y
    
    
