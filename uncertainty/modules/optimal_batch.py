import torch
import math
import numpy as np
from typing import Dict, Callable, Tuple, List
import torch
import torch.nn.functional as F
import pickle

def calculate_MSE(generator: Callable,
                       info_param_1: int = 1.5 * math.pi,
                       info_param_2: int = 1.5 * math.pi,
                       num_samples: int = 10000) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate the accuracy of generated samples for given points.

    Parameters:
    - generator: The model function that generates samples.
    - info_param_1: The first parameter used for generating information.
    - info_param_2: The second parameter used for generating information.
    - num_samples: Number of points to sample.

    Returns:
    - mse1 for the left part of sine and mse2 for the right part of sine.
    """
    
    info1 = torch.linspace(0, info_param_1, num_samples)
    info2 = torch.linspace(info_param_1, info_param_1 + info_param_2, num_samples)

    mse1 = 0.0
    mse2 = 0.0
    for p1, p2 in zip(info1, info2):
        latent_space_sample = torch.randn(1, 1)
        # Generate samples using the generator model
        # print(p1, p2)
        with torch.no_grad():
            res1, res2 = generator(latent_space_sample, p1.repeat(1, 1)), generator(latent_space_sample, p2.repeat(1, 1))
            
        sine_value1, sine_value2  = torch.sin(p1), torch.sin(p2)
        mse1 += (res1.item() - sine_value1.item())**2
        mse2 += (res2.item() - sine_value2.item())**2
        # print(p1, sine_value1, mse1, p2, sine_value2, mse2)
        
    mse1 = mse1/num_samples
    mse2 = mse2/num_samples
    return mse1, mse2



import torch
import math
from typing import Tuple, List

def calculate_metrics(generator,
                      discriminator,
                      info_param_1: float = 1.5 * math.pi,
                      info_param_2: float = 1.5 * math.pi,
                      repeat: int = 20,
                      num_samples: int = 1000) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate the accuracy of generated samples for given points.

    Parameters:
    - generator: The model function that generates samples.
    - discriminator: The discriminator model.
    - info_param_1: The first parameter used for generating information.
    - info_param_2: The second parameter used for generating information.
    - repeat: Number of times to repeat the generation process.
    - num_samples: Number of points to sample.

    Returns:
    - mse1: Mean squared error for the left part of sine.
    - mse2: Mean squared error for the right part of sine.
    - var1: Mean variance for the left part of sine.
    - var2: Mean variance for the right part of sine.
    - entropy1: Entropy for the left part of sine.
    - entropy2: Entropy for the right part of sine.
    """
    
    info1 = torch.linspace(0, info_param_1, num_samples)
    info2 = torch.linspace(info_param_1, info_param_1 + info_param_2, num_samples)

    mse1 = 0.0
    mse2 = 0.0
    var_1 = 0.0
    var_2 = 0.0
    entropy1 = 0.0
    entropy2 = 0.0
    latent_space_samples = torch.randn(repeat, 1)  # Pluralize variable name

    for p1, p2 in zip(info1, info2):
        # Generate samples using the generator model
        with torch.no_grad():
            res1 = generator(latent_space_samples, torch.tensor([[p1.item()]] * repeat))
            res2 = generator(latent_space_samples, torch.tensor([[p2.item()]] * repeat))
            
        sine_value1 = torch.sin(p1)
        sine_value2 = torch.sin(p2)
        mse1 += torch.mean((res1.squeeze() - sine_value1)**2).item()
        mse2 += torch.mean((res2.squeeze() - sine_value2)**2).item()
        
        var_1 += torch.var(res1.squeeze()).item()  
        var_2 += torch.var(res2.squeeze()).item() 
        
        # Calculate discriminator's output and entropy
        with torch.no_grad():
            discriminator_output1 = discriminator(res1, torch.tensor([[p1.item()]] * repeat))
            discriminator_output2 = discriminator(res2, torch.tensor([[p2.item()]] * repeat))
            entropy1 += -torch.mean(discriminator_output1 * torch.log(discriminator_output1 + 1e-8) + \
                                    (1 - discriminator_output1) * torch.log(1 - discriminator_output1 + 1e-8)).item()
            entropy2 += -torch.mean(discriminator_output2 * torch.log(discriminator_output2 + 1e-8) + \
                                    (1 - discriminator_output2) * torch.log(1 - discriminator_output2 + 1e-8)).item()
            
    # Calculate means
    mse1 /= num_samples
    mse2 /= num_samples
    var_1 /= num_samples
    var_2 /= num_samples
    entropy1 /= num_samples
    entropy2 /= num_samples
    
    return mse1, mse2, var_1, var_2, entropy1, entropy2




def append_and_save_mse(generator: Callable,
                        file_path: str,
                        info_param_1: int = 1.5 * math.pi,
                        info_param_2: int = 1.5 * math.pi,
                        num_samples: int = 10000,
                        ) -> None:
    """
    Calculate the MSE of generated samples for given parameters,
    append it to the existing list or create a new one,
    and save it as a pickle file.

    Parameters:
    - generator: The model function that generates samples.
    - info_param_1: The first parameter used for generating information.
    - info_param_2: The second parameter used for generating information.
    - num_samples: Number of points to sample.
    - file_path: Path to the pickle file to save or load the MSE list.
    """
    # Calculate the MSE
    mse1, mse2 = calculate_MSE(generator, info_param_1, info_param_2, num_samples)
    new_mse = (mse1, mse2)
    
    # Try loading the existing G_mse from the pickle file
    try:
        with open(file_path, 'rb') as f:
            G_mse = pickle.load(f)
    except FileNotFoundError:
        G_mse = []

    # Append the new MSE to the list
    G_mse.append(new_mse)

    # Save the updated G_mse as a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(G_mse, f)
    
        

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

def variance4data(generator,
                  data_loader,
                  repeat: int = 20,
                  device='cpu'):
    variances = {}
    latent_space_samples = torch.randn(repeat, 1)
    for batch_idx, (x, info) in enumerate(data_loader):
        batch_size = x.size(0)
        x, info = x.to(device), info.to(device)
        for el in info:
            with torch.no_grad():
                res = generator(latent_space_samples, el.repeat(repeat, 1))
                variances[el.item()] = torch.var(res).item()
                
    def variances2weights(variances: dict):
        """
        Convert dict of variances to dict of weights,
        so, the max weight is 1 for max value of variances and 0 if variance is 0
        """
        max_variance = max(variances.values())
        weights = {key: max(0.01, value / max_variance) for key, value in variances.items()} 
        # print(weights)
        return weights


    return variances2weights(variances)
    

                       
    
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
    
    
    
class WeightedVarianceCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        # super(WeightedIntervalCrossEntropyLoss, self).__init__()
        super().__init__()

    def weights4batch(self, weights: Dict[int, float], conditional_info: torch.Tensor) -> torch.Tensor:

        """
        Assign weights for every point based on conditional information and bins.

        Args:
        - weights (dict): Weights for every point.
        - conditional_info (torch.Tensor): Tensor with conditional information (coordinate x of sine).

        Returns:
        - torch.tensor: Weights for every point in batch that we have in condition.
        """
        batch_weights = []
        for el in conditional_info:
            # print(el)
            batch_weights.append(weights[el.item()])
            
        return torch.tensor(batch_weights)

    def forward(self, y_pred, y_true, weights, conditional_info, from_logits=False):
        weights = self.weights4batch(weights,
                                     conditional_info)

        if from_logits:
            y_pred = F.softmax(y_pred, dim=-1)

        y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
        try:
            loss = F.binary_cross_entropy(y_pred, y_true, weight=weights.unsqueeze(-1))
        except Exception as e:
            print(f'y_pred: {y_pred}, y_true: {y_true}, weights: {weights}')
            print(e)
            raise
            

        return loss