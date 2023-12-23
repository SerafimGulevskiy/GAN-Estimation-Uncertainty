import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import torchvision

from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import os
import numpy as np

from .optimal_batch import calculate_variance


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
    
    
def D_train(
        x,
        info,
        D,
        G,
        D_optimizer,
        criterion,
        device,
        space_dimension = 1,
        noise_dim = 1
):
    D.zero_grad()

    # Train discriminator on real data
    x_real, y_real = x.view(-1, space_dimension).to(device), torch.ones(x.size(0), 1).to(device)  # 1 is real
    # print(x_real, y_real, info)
    D_output = D(x_real, info.view(-1, 1))
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # Train discriminator on fake data
    with torch.no_grad():
        z = torch.randn(x.size(0), noise_dim).to(device)
        fake_info = 2 * math.pi * torch.rand(x.size(0)).view(-1, 1).to(device)
    x_fake, y_fake = G(z, fake_info).view(-1, space_dimension), torch.zeros(x.size(0), 1).to(device)  # 0 is fake
    D_output = D(x_fake, fake_info)
    D_fake_loss = criterion(D_output, y_fake)#train discriminator to find fake results 
    # print(D_output, y_fake)
    D_fake_score = D_output

    # Calculate the total discriminator loss
    D_loss = D_real_loss + D_fake_loss

    # Backpropagate and optimize ONLY D's parameters
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()


def G_train(
        x,
        D,
        G,
        G_optimizer,
        criterion,
        device,
        space_dimension = 1,
        noise_dim = 1
):
    G.zero_grad()
    z = torch.randn(x.size(0), noise_dim).to(device)
    y = torch.ones(x.size(0), 1).to(device)
    
    fake_info = 2 * math.pi * torch.rand(x.size(0)).view(-1, 1).to(device)
    G_output = G(z, fake_info).view(-1, space_dimension)
    D_output = D(G_output, fake_info)
    
    G_loss = criterion(D_output, y.to(device))
    print(D_output, y)
    print(G_loss, G_loss.size())
    print(D_output.size(), y.size())
    print(D_output[0], D_output[0].size())
    print(0/0)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()



def train_epoch(data_loader, D, G, D_optimizer, G_optimizer, criterion, device):
    total_D_loss = 0.0
    total_G_loss = 0.0
    total_samples = 0
    
    for batch_idx, (x, info) in enumerate(data_loader):
        batch_size = x.size(0)
        x, info = x.to(device), info.to(device)
        
        # Train discriminator
        D_loss = D_train(x, info, D, G, D_optimizer, criterion, device)
        
        # Train generator
        G_loss = G_train(x, D, G, G_optimizer, criterion, device)
        
        # Update total losses
        total_D_loss += D_loss * batch_size
        total_G_loss += G_loss * batch_size
        total_samples += batch_size
        
    # Calculate average loss over all samples
    avg_D_loss = total_D_loss / total_samples
    avg_G_loss = total_G_loss / total_samples

    return avg_D_loss, avg_G_loss

def train_epoch_optimal(data_loader, D, G, D_optimizer, G_optimizer, criterion, device):
    
    
    total_D_loss = 0.0
    total_G_loss = 0.0
    total_samples = 0
    
    for batch_idx, (x, info) in enumerate(data_loader):
        batch_size = x.size(0)
        x, info = x.to(device), info.to(device)
        
        # Train discriminator
        D_loss = D_train(x, info, D, G, D_optimizer, criterion, device)
        
        # Train generator
        G_loss = G_train(x, D, G, G_optimizer, criterion, device)
        
        # Update total losses
        total_D_loss += D_loss * batch_size
        total_G_loss += G_loss * batch_size
        total_samples += batch_size
        
    # Calculate average loss over all samples
    avg_D_loss = total_D_loss / total_samples
    avg_G_loss = total_G_loss / total_samples

    return avg_D_loss, avg_G_loss

def train(num_epochs, data_loader, D, G, D_optimizer, G_optimizer, criterion, device, plot_process = False, save_path = None, name = "generated_plots.png"):
    D_losses_final = []
    G_losses_final = []
    Variances = []

    for epoch in tqdm(range(num_epochs)):
        D_loss, G_loss = train_epoch(data_loader,
                    D, G,
                    D_optimizer, G_optimizer,
                    criterion, device)
        
        D_losses_final.append(D_loss)
        G_losses_final.append(G_loss)
        
        b, var = calculate_variance(G, repeat = 10, num_samples = 1000)
        Variances.append(np.mean(var))
        
         
        if epoch % 20 == 0: 
            print(f"epoch [{epoch}/{num_epochs}], average D_loss: {D_loss:.4f}, average G_loss: {G_loss:.4f}")
            
    if plot_process:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(D_losses_final, label='Discriminator')
        plt.plot(G_losses_final, label='Generator')
        plt.title("Training Losses")
        plt.legend()
        
        plt.subplot(1, 2, 2)
            
        plt.plot(Variances, label='Variance')
        plt.title("Training Variance")
        plt.legend()
        
        if save_path:
            save_filename = os.path.join(save_path, name)
            plt.savefig(save_filename)
            
        plt.tight_layout()

        plt.show()
    
        
        
        
            
    return D_losses_final, G_losses_final, Variances

def plot_sine(G, num_samples = 10000, save_path = None, name = "generated_plots.png"):
    
    # First subplot for generated samples
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    
    latent_space_samples = torch.randn(num_samples, 1)

    info = 2 * math.pi * torch.rand(num_samples).view(-1, 1)
    generated_samples = G(latent_space_samples, info)
    
    generated_samples = generated_samples.detach()
    
    plt.plot(info, generated_samples, 'ko', markersize = 0.5)
    plt.xlabel('Info')
    plt.ylabel('Generated Samples')
    plt.title('Generated Samples')
    
    
    # Second subplot for variances
    plt.subplot(1, 2, 2)
    
    points_x, res = calculate_variance(G, repeat = 10, num_samples = num_samples)
    
    # Plot the graph
    plt.plot(points_x, res, 'ko', markersize = 3, label='Variancies')
    plt.xlabel('x')
    plt.ylabel('Model variance')
    plt.title('Model variances at Different Points')
    plt.legend()
    # plt.grid(True)
    # plt.show()
    if save_path is not None:
        # Save the figure in the specified folder
        save_filename = os.path.join(save_path, name)
        plt.savefig(save_filename)
    
    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    plt.show()
    
    
