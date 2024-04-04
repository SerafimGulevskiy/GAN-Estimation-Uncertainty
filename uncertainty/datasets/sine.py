import torch
import math
from torch.utils.data import Dataset

class SinDataset(Dataset):
    def __init__(self, train_data_length=1024, train_data_length_certain=800, multiplier_1=1.5, multiplier_2=2):
        """
        MyDataset class for handling training data.

        Parameters:
        - train_data_length (int): Total length of the training data.
        - train_data_length_certain (int): Length of the certain part of the training data.
        - multiplier_1 (float): Multiplier for the first part of the data.
        - multiplier_2 (float): Multiplier for the second part of the data.
        """
        self.train_data_length = train_data_length
        self.train_data_length_certain = train_data_length_certain
        self.train_data_length_uncertain = train_data_length - train_data_length_certain

        # Initialize the training data tensor directly as an attribute
        self.data = torch.zeros((train_data_length, 1))

    
        certain_values = math.pi * torch.linspace(0, multiplier_1, train_data_length_certain)

        uncertain_values = math.pi * torch.linspace(multiplier_1, multiplier_2, self.train_data_length_uncertain)
        
        # Concatenate the certain and uncertain parts of the data
        self.data[:train_data_length_certain, 0] = certain_values
        self.data[train_data_length_certain:, 0] = uncertain_values

        # Store the sine values of the entire training data
        self.data = [self.data]
        self.data.insert(0, torch.sin(self.data[0][:, 0]))
        
#         # Add complex function to the new part of the data
#         complex_values = torch.linspace(3 * math.pi, 4.5 * math.pi, 974).unsqueeze(1)  # Reshape to have 2 dimensions
#         # complex_function_values = 14 * torch.cos(complex_values) **3 * torch.sin(complex_values) ** 5  # Example complex function
#         complex_function_values = 8 * torch.cos(complex_values) ** 4 * torch.sin(complex_values) ** 6 + torch.sqrt(torch.abs(torch.sin(complex_values))) 
#         complex_function_values = complex_function_values.squeeze(1)

#         # Concatenate the complex function values with existing data
#         self.data[0] = torch.cat((self.data[0], complex_function_values))
#         self.data[1] = torch.cat((self.data[1], complex_values))


    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: A tuple containing input and target values. x is equal coordinate x and y is projection of sine(x) on y
        """
        x = self.data[0][idx]
        y = self.data[1][idx]
        return x, y
    
    
