import torch
import torch.nn.functional as F
from typing import Dict



def normilize_weights_mnist(metrics: dict, min_weight=0.01):
    weights = {i: metrics[i] for i in range(10)}
    max_value = max(metrics.values())
    weights = {k: max(min_weight, v / max_value) for k, v in weights.items()}
    
    return weights

    

class WeightedVarianceBCE(torch.nn.Module):
    def __init__(self):
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
        batch_weights = [weights[el.item()] for el in conditional_info]
        return torch.tensor(batch_weights, device=conditional_info.device)

    def forward(self, y_pred, y_true, weights, conditional_info, from_logits=False):
        if weights:
            weights = self.weights4batch(weights, conditional_info)#it converts into pytorch

        if from_logits:
            y_pred = F.softmax(y_pred, dim=-1)

        y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
        try:
            if weights is not False:
                # print('using weights')
                loss = F.binary_cross_entropy(y_pred, y_true, weight=weights)
            else:
                # print('no weights')
                loss = F.binary_cross_entropy(y_pred, y_true)
        except Exception as e:
            print(f'y_pred: {y_pred}, y_true: {y_true}, weights: {weights}')
            print(e)
            raise

        return loss
