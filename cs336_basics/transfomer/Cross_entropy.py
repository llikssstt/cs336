import torch
import torch.nn as nn
from .softmax import Softmax


def cross_entropy(inputs, targets):

    batch_size = inputs.shape[0]

    max_val = inputs.max(dim=1, keepdim=True).values
    
    inputs_shifted = inputs - max_val
    
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs_shifted), dim=1))
    
    log_probs = inputs_shifted - log_sum_exp.view(batch_size, 1)

    target_log_probs = log_probs[torch.arange(batch_size, device=targets.device), targets]

    return -torch.mean(target_log_probs)