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

from .optimal_batch import calculate_variance, weights_variances
from .animate import plot_sine, plot_training_progress, animated_bar_var_plot, create_gif


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
        noise_dim = 1,
        weights_interval = None
):
    # print(f'D_train size x: {x.size()}')
    # print(z.size(), y.size())
    D.zero_grad()

    # Train discriminator on real data
    x_real, y_real = x.view(-1, space_dimension).to(device), torch.ones(x.size(0), 1).to(device)  # 1 is real
    # print(x_real, y_real, info)
    D_output = D(x_real, info.view(-1, 1))
    # print(f'x info: {info}')
    
    if weights_interval:
        # weights = weights_variancies(G)
        D_real_loss = criterion(D_output, y_real, weights_interval, conditional_info = info)
        #conditional_info need here to calculate weights
        
    else:
        D_real_loss = criterion(D_output, y_real)
    # D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # Train discriminator on fake data
    with torch.no_grad():
        z = torch.randn(x.size(0), noise_dim).to(device)
        # fake_info = 2 * math.pi * torch.rand(x.size(0)).view(-1, 1).to(device)
        
    x_fake, y_fake = G(z, info.view(-1, 1)).view(-1, space_dimension), torch.zeros(x.size(0), 1).to(device)  # 0 is fake 
        # x_fake, y_fake = G(z, fake_info).view(-1, space_dimension), torch.zeros(x.size(0), 1).to(device)  # 0 is fake
    D_output = D(x_fake, info.view(-1, 1))
    # D_output = D(x_fake, fake_info)
    # print(f'D_output: {D_output}, y_fake: {y_fake}')
    
    if weights_interval:
        D_fake_loss = criterion(D_output, y_fake, weights_interval, conditional_info = info)#train discriminator to find fake results 
    else:
        D_fake_loss = criterion(D_output, y_fake)#train discriminator to find fake results 
    # print(D_output, y_fake)
    D_fake_score = D_output
    # print(D_real_loss, D_fake_loss)

    # Calculate the total discriminator loss
    D_loss = (D_real_loss + D_fake_loss)/2
    # print(f'D_real_loss: {D_real_loss}, D_fake_loss: {D_fake_loss}')
    # print(f'D_loss: {D_loss}')

    # Backpropagate and optimize ONLY D's parameters
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()


def G_train(
        x,
        info,
        D,
        G,
        G_optimizer,
        criterion,
        device,
        space_dimension = 1,
        noise_dim = 1,
        weights_interval = None
):
    # print(f'G_train size x: {x.size()}')
    G.zero_grad()
    z = torch.randn(x.size(0), noise_dim).to(device)
    y = torch.ones(x.size(0), 1).to(device)
    # print(z.size(), y.size())
    
    # fake_info = 2 * math.pi * torch.rand(x.size(0)).view(-1, 1).to(device)#fake info maybe is crooectly to take from data or from the same distribution, with the same density
    G_output = G(z, info.view(-1, 1)).view(-1, space_dimension)
    
    D_output = D(G_output, info.view(-1, 1))
    # D_output = D(G_output, fake_info)
    if weights_interval:
        # weights = weights_variancies(G)
        # print(f'weights_interval: {weights_interval}')
        G_loss = criterion(D_output, y.to(device), weights = weights_interval, conditional_info = info.view(-1, 1))
        # G_loss = criterion(D_output, y.to(device), weights = weights_interval, conditional_info = fake_info)
        
    else:
        G_loss = criterion(D_output, y.to(device))
   
    
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

def train_epoch_optimal(data_loader, D, G, D_optimizer, G_optimizer, criterion, device, n_split, weights_interval = False, scheduler_D=None, scheduler_G=None):
    D.eval()
    G.eval()
    # if weights_interval:
    
    weights = weights_variances(G, num_beans = n_split)
    # print(weights)
    
    # print(weights)
    total_D_loss = 0.0
    total_G_loss = 0.0
    total_samples = 0
    D.train()
    G.train()
    
    for batch_idx, (x, info) in enumerate(data_loader):
        batch_size = x.size(0)
        x, info = x.to(device), info.to(device)
        # D_loss = D_train(x, info, D, G, D_optimizer, criterion, device, weights_interval = weights_interval)#weights_interval
        # # Train generator
        # G_loss = G_train(x, D, G, G_optimizer, criterion, device, weights_interval = weights_interval)#weights_interval
        
#         # Train discriminator
#         D_loss = D_train(x, info, D, G, D_optimizer, criterion, device, weights_interval = weights)#weights_interval

#         # Train generator
#         G_loss = G_train(x, info, D, G, G_optimizer, criterion, device, weights_interval = weights)#weights_interval
            
        if weights_interval:
            # Train discriminator
            D_loss = D_train(x, info, D, G, D_optimizer, criterion, device, weights_interval = weights)#weights_interval

            # Train generator
            G_loss = G_train(x, info, D, G, G_optimizer, criterion, device, weights_interval = weights)#weights_interval
            
        else:
            # Train discriminator
            D_loss = D_train(x, info, D, G, D_optimizer, criterion, device)

            # Train generator
            G_loss = G_train(x, info, D, G, G_optimizer, criterion, device)
        
        # Step the schedulers if provided
        if scheduler_D:
            scheduler_D.step()
        if scheduler_G:
            scheduler_G.step()
        
        # Update total losses
        total_D_loss += D_loss * batch_size
        total_G_loss += G_loss * batch_size
        total_samples += batch_size
        
    # Calculate average loss over all samples
    avg_D_loss = total_D_loss / total_samples
    avg_G_loss = total_G_loss / total_samples
    
    # # Step the schedulers if provided
    # if scheduler_D:
    #     scheduler_D.step()
    # if scheduler_G:
    #     scheduler_G.step()

    return avg_D_loss, avg_G_loss, weights

def train(num_epochs,                  # Number of training epochs
          data_loader,                 # DataLoader providing training data
          D, G,                        # Discriminator (D) and Generator (G) models
          D_optimizer, G_optimizer,    # Optimizers for D and G
          criterion,                   # Loss function criterion
          device,                      # Device to perform training on (e.g., 'cuda' or 'cpu')
          plot_process=False,          # Whether to plot the training process
          save_path=None,              # Path to save the plots (if plotting is enabled)
          name="generated_plots.png",  # Name of the saved plot file
          weights_interval=False,      # Whether to use weights for training (optional)
          # plot_info = False,           #save or not variance graphs
          animate_bar_var = False,     #save or not variance bar
          progress_generator = False,  #plot the result of generator every n epoch(fixed 20)
          info_n = 20,                 #write info(metrics, vars and etc.) every info_n epoch
          n_split = 10,                #number of splits
          scheduler_D=None,            # Scheduler for Discriminator optimizer (optional)
          scheduler_G=None,            # Scheduler for Generator optimizer (optional)
         ):
    """
    Returns:
    - D_losses_final (list): List of Discriminator losses for each epoch.
    - G_losses_final (list): List of Generator losses for each epoch.
    - Variances (list): List of variances during training (if applicable).
    """
    D_losses_final = []
    G_losses_final = []
    Variances = []
    weights_var = {}
    create_folder(save_path, name)

    for epoch in tqdm(range(num_epochs)):
        # print(f'weights_interval: {weights_interval}')
        D_loss, G_loss, weights = train_epoch_optimal(data_loader,
                    D, G,
                    D_optimizer, G_optimizer,
                    criterion, device, n_split, weights_interval,
                    scheduler_D=scheduler_D, scheduler_G=scheduler_G)
        D.eval()
        G.eval()
        D_losses_final.append(D_loss)
        G_losses_final.append(G_loss)
        w2i = {v: weights[0][k] for k, v in weights[1].items()}
        v2i = {v: weights[2][k] for k, v in weights[1].items()}
        """
        going through by bins and get values of variances(or values of weights)
        for theese bins(inetrvals) and save it as
        weights_var to make gif after all training
        """
        # print(f'w2i and v2i: {w2i}, {v2i}')
#         for k, v in v2i.items():#for k, v in w2i.items(): if you want to check weights, put w2i
#             if k not in weights_var:
#                 weights_var[k] = []
            
#             weights_var[k].append(v)
        # print(weights_var)
        
        # b, var, mean, generated = calculate_variance(G, repeat = 10, num_samples = 1000)
        
        result = calculate_variance(G,
                                 info_param_1 = 3 * math.pi,
                                 info_param_2 = 0,
                                 mean = False)
        
        b, generated, var = zip(*[(k, el['G'], el['variance']) for k, el in result.items()])
            
        # Variances.append(np.mean(var))
        Variances.append(np.max(var))
        
         
        if epoch % info_n == 0: 
            print(f"epoch [{epoch}/{num_epochs}], average D_loss: {D_loss:.4f}, average G_loss: {G_loss:.4f}")
            if progress_generator:
                plot_sine(G, save_path=save_path, name = name, epoch = epoch);
            if animate_bar_var:
                animated_bar_var_plot(weights_variance = v2i, epoch = epoch, save_path=save_path, name = name, weights_bins = w2i)
            
    if plot_process:
        plot_training_progress(D_losses_final, G_losses_final, Variances, save_path = save_path, name = name);
        
#     if animate_bar_var:
#         animated_bar_var_plot(weights_var, save_path=save_path, name = name);
#         pass
        
    if progress_generator:
        plot_sine(G, save_path=save_path, name = name, epoch = epoch + 1);
        file_paths = [os.path.join(save_path, name, f'generated_plots_epoch_{epoch}.png') for epoch in range(0, num_epochs + info_n, info_n)]
        create_gif(file_paths, save_path = save_path, name = name, duration = num_epochs, gif_path='final_generated_plots');
        
        animated_bar_var_plot(weights_variance = v2i, epoch = epoch + 1, save_path=save_path, name = name, weights_bins = w2i)
        file_paths = [os.path.join(save_path, name, f'bar_var_{epoch}.png') for epoch in range(0, num_epochs + info_n, info_n)]
        create_gif(file_paths, save_path = save_path, name = name, duration = num_epochs, gif_path='final_bar_vars');
        
        file_paths = [os.path.join(save_path, name, f'bar_weight_{epoch}.png') for epoch in range(0, num_epochs + info_n, info_n)]
        create_gif(file_paths, save_path = save_path, name = name, duration = num_epochs, gif_path='final_bar_weights');
        
        
        
        
    
        
        
    

            
    return D_losses_final, G_losses_final, Variances, weights_var
    
def create_folder(base_path, folder_name):  
    full_path = os.path.join(base_path, folder_name)
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Folder '{folder_name}' created at '{base_path}'.")
    else:
        pass


