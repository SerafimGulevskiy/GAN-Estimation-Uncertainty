import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import torch
import math
import numpy as np
from .optimal_batch import calculate_variance, weights_variances
    
    
def animated_bar_var_plot(weights_variance: dict, epoch:int, save_path = None, name = None, weights_bins:dict = None):
    title="Animated Bar Plot"
    categories, values_list = zip(*weights_variance.items())
    plt.bar(categories, values_list, edgecolor='r')
  
    plt.ylim(0, 1)
    plt.ylabel('Variance')
    plt.title(f'Variances, epoch {epoch}')
    
    
    if save_path:
        save_filename = os.path.join(save_path, name, f'bar_var_{epoch}.png')
        plt.savefig(save_filename, dpi=300)
    plt.close()
    
    if weights_bins:
        categories, weights_list = zip(*weights_bins.items())
        # v1, v2 = [], []
        # for i in range(len(weights_list)):
        #     v1.append(max(weights_list[i], values_list[i]))
        #     v2.append(min(weights_list[i], values_list[i]))
        # Create weights bars
        plt.bar(categories, [0.01] * len(weights_list), color='b')#edgecolor='purple'
        plt.bar(categories, weights_list, color='lightblue')#, edgecolor='r'
        plt.legend(labels=['Added', 'Weights'])
        plt.ylim(0, 1)
        plt.ylabel('Weight')
        plt.title(f'Weights, epoch {epoch}')
    
        if save_path:
            save_filename = os.path.join(save_path, name, f'bar_weight_{epoch}.png')
            plt.savefig(save_filename, dpi=300)
        plt.close()
    
# def animated_bar_var_plot(weights_variance: dict, title="Animated Bar Plot", ylim=(0, 1), figsize=(8, 6), save_path = None, name = None):
    
#     categories, values_list = zip(*weights_variance.items())
#     print(type(categories), type(values_list))
#     # return categories, values_list
    
#     fig, axes = plt.subplots(figsize=figsize)
#     axes.set_ylim(ylim)

#     bars = axes.bar(categories, [el[0] for el in values_list])

#     def animate(i):
#         for j in range(len(categories)):
#             bars[j].set_height(values_list[j][i])

#     plt.title(title, fontsize=14)
#     ani = FuncAnimation(fig, animate, frames=len(values_list[0]), repeat=False)
    
#     # Save animation as GIF
#     if save_path:
#         save_filename = os.path.join(save_path, name, 'bar_var.gif')
#         ani.save(save_filename, writer='imagemagick', fps=2)  # You need to have imagemagick installed for this to work
#         plt.close(fig)  # Close the figure without displaying it

#     # return HTML(ani.to_jshtml())

def plot_training_progress(D_losses, G_losses, variances, save_path, name):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(D_losses, label='Discriminator')
    plt.plot(G_losses, label='Generator')
    plt.title("Training Losses")
    plt.legend()
    
    if variances:
        plt.subplot(1, 2, 2)
        plt.plot(variances, label='Variance')
        plt.title("Training Variance")
        plt.legend()

    if save_path:
        save_filename = os.path.join(save_path, name,  'process')
        plt.savefig(save_filename, dpi=300)

    plt.tight_layout()
    plt.close()
    # plt.show();
    
def plot_sine(G, epoch, name, num_samples = 10000, save_path = None):
    
    # First subplot for generated samples
    plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    
    latent_space_samples = torch.randn(num_samples, 1)
    with torch.no_grad():
        result = calculate_variance(G,
                                             info_param_1 = 6 * math.pi,
                                             info_param_2 = 2 * math.pi,
                                             repeat = 20,
                                             num_samples = 1000)


    info, generated, mean, var = zip(*[(k, el['G'], el['mean'], el['variance']) for k, el in result.items()])
    
    # info, var, mean, generated = calculate_variance(G,
    #                                      info_param_1 = 6 * math.pi,
    #                                      info_param_2 = 2 * math.pi,
    #                                      repeat = 20,
    #                                      num_samples = 1000)
    # return info, var, mean, generated
    # p1 = [min(el) for el in generated]
    # p2 = [max(el) for el in generated]
    # p1 = [np.mean(el) - np.var(el) for i, el in enumerate(generated)]
    # p2 = [np.mean(el) + np.var(el) for i, el in enumerate(generated)]
    p1 = [m - v for m, v in zip(mean, var)]
    p2 = [m + v for m, v in zip(mean, var)]
    
    # print(np.var(p1[0]))
    
    info = torch.tensor(info)
    info, _ = info.sort(dim=0)
    
    # plt.plot(info, generated, 'ko', markersize = 0.05)
    plt.xlabel('Info')
    plt.ylabel('Generated Samples')
    plt.title(f'Generated Samples, epoch {epoch}')
    x_values = np.linspace(0, 2 * np.pi, 100)
    sin_values = np.sin(info)
    plt.plot(info, sin_values.squeeze(), label='sin(x)', linestyle='-', linewidth=2, color = 'r') 

    plt.plot(info, mean, label='mean(G)', linestyle='-', linewidth=2, color = 'b')
    plt.fill_between(info, p1, p2, alpha=0.9)
    
    if save_path is not None:
        # Save the figure in the specified folder
        save_filename = os.path.join(save_path, name, f'generated_plots_epoch_{epoch}.png')
        # save_filename = os.path.join(save_path, name, 'generated_plots.png')
        plt.savefig(save_filename, dpi=300)
    plt.close()
        
    # plt.show()
    
def create_gif(file_paths, gif_path, save_path, name, duration, loop=0):
    """
    Create a GIF animation from a list of image files.

    Parameters:
        file_paths (list): List of file paths for the images.
        gif_path (str): Path to save the GIF file.
        save_path (str): Directory to save the GIF file.
        name (str): Name of the GIF file (without extension).
        duration (int): Duration (in milliseconds) of each frame in the GIF.
        loop (int, optional): Number of loops for the GIF (0 for infinite loop).
        
    Returns:
        None
    """
        
    images = [plt.imread(file_path) for file_path in file_paths]
    
    plt.figure(figsize=(12, 6))
    plt.axis('off')
    ims = plt.imshow(images[0])

    def update(i):
        ims.set_array(images[i])
        # plt.text(0.5, 0.5, str(i), fontsize=24, color='white', ha='center', va='center')  # Overlay number

    ani = FuncAnimation(plt.gcf(), update, frames=len(images), interval=duration, repeat_delay=1000)
    
    gif_path = os.path.join(save_path, name, f'{gif_path}.gif')
    ani.save(gif_path, writer='imagemagick', fps=2, dpi=300)
    
    # Delete all files except the last epoch
    for file_path in file_paths[:-1]:
        os.remove(file_path)
    plt.close()
    
#     # Second subplot for variances
#     plt.subplot(1, 2, 2)
#     points_x, res = calculate_variance(G, repeat = 10, num_samples = num_samples)
    
#     # Plot the graph
#     plt.plot(points_x, res, 'ko', markersize = 3, label='Variancies')
#     plt.xlabel('x')
#     plt.ylabel('Model variance')
#     plt.title('Model variances at Different Points')
#     plt.legend()
    
#     if save_path is not None:
#         # Save the figure in the specified folder
#         save_filename = os.path.join(save_path, name, 'generated_plots.png')
#         plt.savefig(save_filename)
    
#     # Adjust layout to prevent clipping of titles
#     plt.tight_layout()

