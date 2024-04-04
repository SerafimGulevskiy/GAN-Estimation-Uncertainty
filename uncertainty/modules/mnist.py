import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
from tqdm import tqdm

from .mnist_animate import generate_and_save_fake_image_grid, plot_training_progress, plot_training_fid
from .animate import create_gif
from .mnist_classifier import calculate_confusion_matrix

from .mnist_classifier import eval_model

LATENT_DIM = 100

class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, image_shape=(1, 28, 28)):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        
        self.label_embedding = nn.Embedding(num_embeddings = 10, embedding_dim = 10)#first param is num_embeddings means number of claases, for every class we train embeddings, embedding_dim is merely a dimension of embedding
        
        def block(in_features, out_features, normalize=False):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            # layers.append(nn.GELU())
            return layers
    
    
        self.model = nn.Sequential(
            *block(self.latent_dim + 10, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            # *block(512, 1024),
            nn.Linear(1024, 784),
            nn.Tanh()
        )


    def forward(self, noise, labels):
        labels_embs = self.label_embedding(labels)
        x = torch.cat([noise, labels_embs], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)
    
    
# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, image_shape=(1, 28, 28)):
        super().__init__()
        self.image_shape = image_shape
        self.label_embedding = nn.Embedding(10, 10)
        
        def block(in_features, out_features, dropout = 0.3):
            layers = [nn.Linear(in_features, out_features)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            # layers.append(nn.GELU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            
            return layers
        
        self.model = nn.Sequential(
            *block(10 + int(np.prod(self.image_shape)), 512),
            *block(512, 512),
            *block(512, 512),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )


    def forward(self, img, labels):
        x = img.view(img.size(0), -1)
        labels_embs = self.label_embedding(labels)
        x = torch.cat([x, labels_embs], 1)
        out = self.model(x)
        return out.squeeze()
    
    
def D_train(
        x,
        info,
        D,
        G,
        D_optimizer,
        criterion,
        device,
        space_dimension = 1,
        noise_dim = 100,
        weights_interval = None
):

    D.zero_grad()
    # Train discriminator on real data
    x_real, y_real = x.to(device), torch.ones(x.size(0)).to(device)  # 1 is real
    
    # print(x_real, y_real, info)
    D_output = D(x_real, info)
    # print(x_real.size(), info.size(), y_real.size())
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # Train discriminator on fake data
    with torch.no_grad():
        z = torch.randn(x.size(0), noise_dim).to(device)
        # print(f'z: {z.size()}, info: {info.size()}, noise: noise_dim')
        # fake_info = 2 * math.pi * torch.rand(x.size(0)).view(-1, 1).to(device)
        x_fake, y_fake = G(z, info).to(device), torch.zeros(x.size(0)).to(device)# 0 is fake

    D_output = D(x_fake, info)

    D_fake_loss = criterion(D_output, y_fake)#train discriminator to find fake results 
    D_fake_score = D_output

    # Calculate the total discriminator loss
    D_loss = (D_real_loss + D_fake_loss)/2

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
        noise_dim = 100,
        weights_interval = None
):
    # print(f'G_train size x: {x.size()}')
    G.zero_grad()
    z = torch.randn(x.size(0), noise_dim).to(device)
    y = torch.ones(x.size(0)).to(device)
    
    G_output = G(z, info).to(device)
    
    D_output = D(G_output, info)
    G_loss = criterion(D_output, y.to(device))
    
    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()



def train_epoch_optimal(data_loader, D, G, D_optimizer, G_optimizer, criterion, device, n_split, weights_interval = False, scheduler_D=None, scheduler_G=None):
    # D.eval()
    # G.eval()
    # if weights_interval:
    
    # weights = weights_variances(G, num_beans = n_split)
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


    return avg_D_loss, avg_G_loss



def train(num_epochs,                  # Number of training epochs
          data_loader,                 # DataLoader providing training data
          D, G,                        # Discriminator (D) and Generator (G) models
          D_optimizer, G_optimizer,    # Optimizers for D and G
          criterion,                   # Loss function criterion
          device,                      # Device to perform training on (e.g., 'cuda' or 'cpu')
          plot_process = False,          # Whether to plot the training process
          save_path = None,              # Path to save the plots (if plotting is enabled)
          name = "generated_plots.png",  # Name of the saved plot file
          weights_interval = False,      # Whether to use weights for training (optional)
          # plot_info = False,           #save or not variance graphs
          animate_bar_var = False,     #save or not variance bar
          progress_generator = False,  #plot the result of generator every n epoch(fixed 20)
          info_n = 20,                 #write info(metrics, vars and etc.) every info_n epoch
          n_split = 10,                #number of splits
          scheduler_D = None,            # Scheduler for Discriminator optimizer (optional)
          scheduler_G = None,            # Scheduler for Generator optimizer (optional)
          classifier = None,            #classifier to watch its accuracy and fid over the training
          accuracy = None,
          fid = None,
          fid_dataset = None
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
    classifier_res = {'loss_CE': [], 'accuracy': [], 'FID_cats': [], 'vFID_cats': []}
    weights_var = {}
    
    
    if save_path:
        create_folder(save_path, name)
    
    if fid and classifier and fid_dataset:
        from .fid import split_mnist_cats, calculate_multiple_fid
        category_data_real = split_mnist_cats(fid_dataset)

    for epoch in tqdm(range(num_epochs)):
        # print(f'weights_interval: {weights_interval}')
        D_loss, G_loss = train_epoch_optimal(data_loader,
                    D, G,
                    D_optimizer, G_optimizer,
                    criterion, device, n_split, weights_interval,
                    scheduler_D=scheduler_D, scheduler_G=scheduler_G)
        D.eval()
        G.eval()
        D_losses_final.append(D_loss)
        G_losses_final.append(G_loss)
        if classifier:
            fake_loader = get_fake_dataloader(G,
                                              device,
                                              batch_size=32,
                                              num_examples_per_class=1000,
                                              noise_dim=LATENT_DIM,
                                              shuffle=True)
            
            loss_test, accuracy_test = eval_model(fake_loader,
                                      classifier,
                                      criterion = nn.CrossEntropyLoss(),
                                      device = device)
            
            classifier_res['loss_CE'].append(loss_test)
            classifier_res['accuracy'].append(accuracy_test)
            if fid and classifier and fid_dataset:
                fid_cats, vfid_cats = calculate_multiple_fid(G, classifier, category_data_real, device)
                # print(fid_cats, vfid_cats)
                classifier_res['FID_cats'].append(fid_cats)
                classifier_res['vFID_cats'].append(vfid_cats)
# print(loss_test, accuracy_test)
            
            
        # w2i = {v: weights[0][k] for k, v in weights[1].items()}
        # v2i = {v: weights[2][k] for k, v in weights[1].items()}
        """
        going through by bins and get values of variances(or values of weights)
        for theese bins(inetrvals) and save it as
        weights_var to make gif after all training
        """
        if epoch % info_n == 0: 
            print(f"epoch [{epoch}/{num_epochs}], average D_loss: {D_loss:.4f}, average G_loss: {G_loss:.4f}")
            if classifier:
                print(f"calssifier: loss CE -- {loss_test}, accuracy -- {accuracy_test}")
                
                calculate_confusion_matrix(classifier,
                                           fake_loader,
                                           device,
                                           epoch,
                                           save_path = save_path,
                                           name = name)
            if progress_generator:
                generate_and_save_fake_image_grid(G, save_path=save_path, name = name, epoch = epoch);
                
    # if save_path:
    #     generate_and_save_fake_image_grid(G, save_path=save_path, name = name, epoch = epoch);
        # print(f'w2i and v2i: {w2i}, {v2i}')
#         for k, v in v2i.items():#for k, v in w2i.items(): if you want to check weights, put w2i
#             if k not in weights_var:
#                 weights_var[k] = []
            
#             weights_var[k].append(v)
        # print(weights_var)
        
        # b, var, mean, generated = calculate_variance(G, repeat = 10, num_samples = 1000)
        

        
        # b, generated, var = zip(*[(k, el['G'], el['variance']) for k, el in result.items()])
            
        # Variances.append(np.mean(var))
        # Variances.append(np.max(var))
        
    if plot_process:
        plot_training_progress(D_losses_final,
                               G_losses_final,
                               Variances,
                               classifier_res,
                               save_path = save_path,
                               name = name);
        if fid and classifier and fid_dataset:
            print('classifier_res', classifier_res)
            plot_training_fid( classifier_res,
                               save_path = save_path,
                               name = name)
        
    if progress_generator:
        
        generate_and_save_fake_image_grid(G,
                                          save_path=save_path,
                                          name = name,
                                          epoch = epoch + 1);
        file_paths = [os.path.join(save_path, name, f'fake_images_{epoch}.png') for epoch in range(0, num_epochs + info_n, info_n)]
        create_gif(file_paths, save_path = save_path, name = name, duration = num_epochs, gif_path='generated_images_grid');
        
    if classifier:
        calculate_confusion_matrix(classifier,
                                           fake_loader,
                                           device,
                                           epoch = epoch + 1,
                                           save_path = save_path,
                                           name = name)
        file_paths = [os.path.join(save_path, name, f'cm_epoch_{epoch}.png') for epoch in range(0, num_epochs + info_n, info_n)]
        create_gif(file_paths, save_path = save_path, name = name, duration = num_epochs, gif_path='cm_res_grid');
        

            
    return D_losses_final, G_losses_final
    
def create_folder(base_path, folder_name):  
    full_path = os.path.join(base_path, folder_name)
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Folder '{folder_name}' created at '{base_path}'.")
    else:
        pass

class FakeDataset(Dataset):
    def __init__(self, generator, device, num_examples_per_class=1000, noise_dim=LATENT_DIM):
        self.generator = generator
        self.device = device
        self.num_examples_per_class = num_examples_per_class
        self.noise_dim = noise_dim
        self.data = []
        self.labels = []

        # Function to generate fake samples for a given class
        def generate_fake_samples(num_samples, class_label):
            z = torch.randn(num_samples, self.noise_dim).to(device)
            labels = torch.full((num_samples,), class_label).to(device)
            with torch.no_grad():
                fake_images = self.generator(z, labels).to(device)  # Move images to CPU for compatibility with torchvision datasets
            fake_images = fake_images.view(-1, 1, 28, 28)# Reshape each image to have size (1, 28, 28)
            return fake_images, labels

        # Generate fake dataset for each class
        for class_label in range(10):
            fake_images, labels = generate_fake_samples(self.num_examples_per_class, class_label)
            self.data.append(fake_images)
            self.labels.append(labels)

        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
def get_fake_dataloader(generator, device,
                        batch_size=32, num_examples_per_class=1000,
                        noise_dim=LATENT_DIM, shuffle=True):
    fake_dataset = FakeDataset(generator, device, num_examples_per_class, noise_dim)
    fake_dataloader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=shuffle)
    return fake_dataloader

