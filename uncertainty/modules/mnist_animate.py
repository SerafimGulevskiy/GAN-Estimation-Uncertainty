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
    
    # Save the grid
    save_filename = os.path.join(save_path, name, f'fake_images_{epoch}.png')
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    grid_image = grid.permute(1, 2, 0).cpu().data.numpy()
    plt.imsave(save_filename, grid_image, cmap='binary')