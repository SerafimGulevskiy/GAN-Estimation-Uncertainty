import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import torch
import math
import numpy as np

from torchvision.utils import make_grid
SEED = 42
NUM_IMAGES = 100
NROW = 10
Z_SPACE = 100
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Z_FAKE = torch.randn(NUM_IMAGES, Z_SPACE, device=device)
FIXED_LABELS = torch.tensor([i for i in range(10) for _ in range(NROW)]).cpu()

def generate_and_save_fake_image_grid(G, save_path: str, name: str, epoch: int, z_fake: torch.Tensor = Z_FAKE, fake_labels: torch.Tensor = FIXED_LABELS, num_images=NUM_IMAGES, nrow=NROW):
    """
    Generate fake images using a trained generator network and save them as a grid image.

    Parameters:
        G (torch.nn.Module): Generator network.
        save_path (str): Directory to save the fake images.
        name (str): Name of the dataset or experiment.
        epoch (int): Current epoch or iteration number.
        latent_space_vectors (torch.Tensor): Latent space vectors used for generating images.
        fake_labels (torch.Tensor): Labels for conditional generation.
        num_images (int): Number of fake images to generate.
        nrow (int): Number of images per row in the grid.
    """
        
    G.eval()  # Set generator to evaluation mode
    
    # Generate fake images
    with torch.no_grad():
        fake_imgs = G(z_fake, fake_labels).unsqueeze(1)
    
    # Create grid of fake images
    grid = make_grid(fake_imgs, nrow=nrow, normalize=True)
    
    # Add epoch number to the image
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().data.numpy(), cmap='binary')
    plt.axis('off')
    # plt.text(10, 10, f'Epoch: {epoch}', color='white', fontsize=120, weight='bold')
    
    # Save the grid
    save_filename = os.path.join(save_path, name, f'fake_images_{epoch}.png')
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    grid_image = grid.permute(1, 2, 0).cpu().data.numpy()
    plt.imsave(save_filename, grid_image, cmap='binary')
    
    plt.close() 
    
    
def plot_training_progress(D_losses, G_losses, variances, classifier_res, save_path, name, metrics_name = 'process.png'):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(D_losses, label='Discriminator')
    plt.plot(G_losses, label='Generator')
    plt.title("Training Losses")
    plt.legend()
    
    if classifier_res:
        plt.subplot(1, 2, 2)
        plt.plot(classifier_res['loss_CE'], label='loss_CE')
        plt.plot(classifier_res['accuracy'], label='Accuracy')
        plt.title("pretrained classifier results")
        plt.legend()
        
    if variances:
        plt.subplot(1, 2, 3)
        plt.plot(variances, label='Variance')
        plt.title("Training Variance")
        plt.legend()

    if save_path:
        save_filename = os.path.join(save_path, name,  metrics_name)
        plt.savefig(save_filename, dpi=300)

    plt.tight_layout()
    plt.close()
    
def plot_training_fid(classifier_res, save_path, name):
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for category in range(10):
        values = [d[category] for d in classifier_res['FID_cats']]
        plt.plot(values, label=f'Category {category}')

    plt.title("FID for each category")
    plt.xlabel("Epoch")
    plt.ylabel("FID Score")
    plt.legend()
    # plt.ylim(0, 1000)  # Set y-axis limit
    
    plt.subplot(1, 2, 2)
    
    for category in range(10):
        values = [d[category] for d in classifier_res['vFID_cats']]
        plt.plot(values, label=f'Category {category}')

    plt.title("vFID for each category")
    plt.xlabel("Epoch")
    plt.ylabel("vFID Score")
    plt.legend()
    # plt.ylim(0, 100)  # Set y-axis limit
        

    if save_path:
        save_filename = os.path.join(save_path, name,  'process_fid')
        plt.savefig(save_filename, dpi=300)

    plt.tight_layout()
    plt.close()