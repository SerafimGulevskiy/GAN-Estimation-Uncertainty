import torch
import math
import numpy as np

def calculate_variance(Generator, repeat = 10, num_samples = 1000):
    """
    Calculate variance for points x to get uncertainty of the model.

    Parameters:
    - G: The model function.
    - num_samples: Number of samples.
    - repeat: Number of repeats.

    Returns:
    - Tuple of points and corresponding variances.
    """
    latent_space_samples = torch.randn(repeat, 1)
    info = 2 * math.pi * torch.rand(num_samples)

    result = []
    points = []
    for el in info:

        res = Generator(latent_space_samples, el.repeat(repeat, 1))
        # Calculate the stats: variance 
        # mean_value = torch.mean(res)
        variance = torch.var(res)
        # max_value = torch.max(res)
        # min_value = torch.min(res)
        # difference = max_value - min_value
        result.append(variance.item())
        points.append(el.item())
        
        
    return points, result
        
    
def create_batch(Generator,
                 batch_size:int,
                 get_variance = False,
                 repeat = 10,
                 num_samples = 1000):
    """
    create batch by info about variances,
    add points, where the variance is the biggest
    """
    # random_noise = torch.randn(batch_size)
    
    points, var = calculate_variance(Generator, repeat, num_samples)
    batch_size_ids = np.argsort(var)[-batch_size:]
    points_k = torch.tensor([torch.tensor(points[i]) for i in batch_size_ids])
    points_k = points_k.view(-1, 1)
    targets_k = torch.sin(points_k).view(-1)
    
    batch = [targets_k, points_k]
    if get_variance:
        mean_var = np.mean(var)
        return batch, mean_var
    
    return batch, None


def weights_variancies(G, num_beans = 10):
    points, variances = calculate_variance(G)
    
    bin_edges = np.linspace(0, 2 * np.pi, num_beans + 1)
    bins_dict = {i + 1: bin_edges[i] for i in range(num_beans)}
    bins = np.digitize(points, bin_edges)
    
    data_in_bins = {bin_num: {'points': [], 'variances': []} for bin_num in range(1, num_beans + 1)}
    for i in range(len(variances)):
        bin_num = bins[i]
        data_in_bins[bin_num]['variances'].append(variances[i])
        data_in_bins[bin_num]['points'].append(points[i])
        
    mean_variances_in_bins = {bin_num: np.mean(data['variances']) for bin_num, data in data_in_bins.items()}
    total_sum_var = sum(mean_variances_in_bins.values())
    weights = {bin_num: mean / total_sum_var for bin_num, mean in mean_variances_in_bins.items()}
    
    return weights, bins_dict
    
    

    
    
    