import torch
import math
import numpy as np
from typing import Dict, Callable, Tuple, List
import torch
import torch.nn.functional as F

# def calculate_variance(generator: Callable,
#                        info_param_1: int = 2 * math.pi,
#                        info_param_2: int = 0,
#                        repeat: int = 10, 
#                        num_samples: int = 1000) -> Tuple[List[float], List[float], List[float]]:
#     """
#     Calculate the variance of generated samples for given points.

#     Parameters:
#     - generator: The model function that generates samples.
#     - info_param_1: The first parameter used for generating information.
#     - info_param_2: The second parameter used for generating information.
#     - repeat: Number of repeats for each point.
#     - num_samples: Number of points to sample.

#     Returns:
#     - Tuple of points and corresponding variances.
#     """
#     # Generate random samples in the latent space
#     latent_space_samples = torch.randn(repeat, 1)
    
#     # Generate random information for sampling points
#     # info = info_param_1 * torch.rand(num_samples)
#     info = torch.linspace(0, info_param_1, num_samples)
#     if info_param_2:
#         info -= info_param_2

    
#     result = []
#     points = []
#     result_mean = []
#     all_res = []
    
#     for el in info:
#         # Generate samples using the generator model
#         res = generator(latent_space_samples, el.repeat(repeat, 1))
#         # Calculate the variance of the generated samples
#         variance = torch.var(res)
#         mean = torch.mean(res)
        
#         # Append the variance and corresponding point to the result lists
#         result.append(variance.item())
#         points.append(el.item())
#         result_mean.append(mean.item())
#         all_res.append(res)
        
#     all_res = [el.squeeze().tolist() for el in all_res]
#     # all_res = [item for sublist in all_res for item in sublist]
#     # Extracting lists of points
#     # all_result = []
#     # all_res = all_res.squeeze().tolist()
#     # flattened_list = [item for sublist in all_res for item in sublist]
#     # for tensor in all_res:
#     #     points_0 = tensor.detach().numpy().flatten().tolist()
#     #     all_result.append(points_0)
        
        
#     return points, result, result_mean, all_res

def calculate_variance(generator: Callable,
                       info_param_1: int = 3 * math.pi,
                       info_param_2: int = 0,
                       repeat: int = 20, 
                       num_samples: int = 3000,
                       mean: bool = True,
                       variance: bool = True):
    """
    Calculate the variance of generated samples for given points.

    Parameters:
    - generator: The model function that generates samples.
    - info_param_1: The first parameter used for generating information.
    - info_param_2: The second parameter used for generating information.
    - repeat: Number of repeats for each point.
    - num_samples: Number of points to sample.

    Returns:
    - Dictionary with points as keys and corresponding variances, means, and generated values.
    """
    # Generate random samples in the latent space
    latent_space_samples = torch.randn(repeat, 1)
    
    # Generate random information for sampling points
    # info = info_param_1 * torch.rand(num_samples)
    info = torch.linspace(0, info_param_1, num_samples)
    if info_param_2:
        info -= info_param_2

    result = {}
    
    for el in info:
        result[el.item()] = {}
        # Generate samples using the generator model
        with torch.no_grad():
            res = generator(latent_space_samples, el.repeat(repeat, 1))
        result[el.item()]['G'] = res.squeeze().tolist()
        if mean:
            result[el.item()]['mean'] = torch.mean(res).item()
        if variance:
            # result[el.item()]['variance'] = torch.std(res).item()
            result[el.item()]['variance'] = torch.var(res).item()
        
    return result
        
    
def create_batch(Generator,
                 batch_size:int,
                 get_variance = False,
                 repeat = 20,
                 num_samples = 1000):
    """
    create batch by info about variances,
    add points, where the variance is the biggest
    """
    # random_noise = torch.randn(batch_size)
    
    # points, var, mean, generated = calculate_variance(Generator, repeat, num_samples)
    result = calculate_variance(Generator,
                                 info_param_1 = 2 * math.pi,
                                 info_param_2 = 0,
                                 repeat = repeat,
                                 num_samples = num_samples)


    info, generated, mean, var = zip(*[(k, el['G'], el['mean'], el['variance']) for k, el in result.items()])
    
    batch_size_ids = np.argsort(var)[-batch_size:]
    points_k = torch.tensor([torch.tensor(points[i]) for i in batch_size_ids])
    points_k = points_k.view(-1, 1)
    targets_k = torch.sin(points_k).view(-1)
    
    batch = [targets_k, points_k]
    if get_variance:
        mean_var = np.mean(var)
        return batch, mean_var
    
    return batch, None


def weights_variances(G, num_beans = 10):
    # points, variances, mean, generated = calculate_variance(G)
    with torch.no_grad():
        result = calculate_variance(G,
                                     info_param_1 = 3 * math.pi,
                                     info_param_2 = 0,
                                     mean = False)


    points, generated, variances = zip(*[(k, el['G'], el['variance']) for k, el in result.items()])
    bin_edges = np.linspace(0, 3 * math.pi, num_beans + 1)
    bin_edges[-1] += 0.01
    bins_dict = {i + 1: bin_edges[i] for i in range(num_beans)}
    bins = np.digitize(points, bin_edges)
    
    data_in_bins = {bin_num: {'points': [], 'variances': []} for bin_num in range(1, num_beans + 1)}
    for i in range(len(variances)):
        bin_num = bins[i]

        data_in_bins[bin_num]['variances'].append(variances[i])
        data_in_bins[bin_num]['points'].append(points[i])
        
    mean_variances_in_bins = {bin_num: np.mean(data['variances']) for bin_num, data in data_in_bins.items()}
    total_sum_var = sum(mean_variances_in_bins.values())
    # vars = {bin_num: mean for bin_num, mean in mean_variances_in_bins.items()}
    weights = {bin_num: mean / total_sum_var for bin_num, mean in mean_variances_in_bins.items()}
    
    return weights, bins_dict, mean_variances_in_bins


class WeightedIntervalCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        # super(WeightedIntervalCrossEntropyLoss, self).__init__()
        super().__init__()

    def weights4batch(self, weights: Dict[int, float], conditional_info: torch.Tensor, bins_values: Dict[int, float]) -> torch.Tensor:

        """
        Assign weights for every point based on conditional information and bins.

        Args:
        - weights (dict): Weights for every interval.
        - conditional_info (torch.Tensor): Tensor with conditional information (coordinate x of sine).
        - bins_values (dict): Bins values.

        Returns:
        - torch.tensor: Weights for every point that we have in condition.
        """
        weights_bins = []
        number_bins = len(bins_values.values())
        # min_weight = 1/number_bins
        min_weight = 0.01
        # print(f'conditional_info: {conditional_info}')
        sorted_info, indices = torch.sort(conditional_info, axis = 0)

    
        current_bin_id = 1
        # print(f'sorted_info and indices: {sorted_info}, {indices}')
        for i in range(conditional_info.size(0)):
            #find the next current_bin_id
            while current_bin_id <= number_bins - 1 and not (
                bins_values[current_bin_id] < sorted_info[i].item() < bins_values[current_bin_id + 1]
            ):
                current_bin_id += 1
            weights_bins.append(max(weights[current_bin_id], min_weight))
            # weights_bins.append(weights[current_bin_id])
        
        weights_bins = torch.tensor(weights_bins)
        sorted_weights = weights_bins[torch.argsort(indices.view(-1))]
        # print(sorted_weights, conditional_info)

        return sorted_weights

    def forward(self, y_pred, y_true, weights, conditional_info, from_logits=False):
        weights = self.weights4batch(weights[0],
                                     conditional_info,
                                     bins_values=weights[1])

        if from_logits:
            y_pred = F.softmax(y_pred, dim=-1)

        y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
        try:
            loss = F.binary_cross_entropy(y_pred, y_true, weight=weights.unsqueeze(-1))
        except Exception as e:
            print(f'y_pred: {y_pred}, y_true: {y_true}, weights: {weights.unsqueeze(-1)}')
            print(e)
            raise
            

        return loss