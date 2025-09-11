"""Federated LDP Quantile Regression Implementation."""

from typing import Union
import torch
import torch.nn as nn
import numpy as np
import math
from flwr.common import Context
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
import itertools
_global_data= None

class QuantileNet(nn.Module):
    """Quantile Regression Model"""
    
    def __init__(self, input_dim=6):
        super(QuantileNet, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def generate_data(rounds: int, local_updates_mode: Union[str|int], n_clients: int):
    """Generate synthetic data for quantile regression."""

    # Set the Em of the top 5% rounds to 1.
    Em_list = []
    first_5_percent = int(rounds * 0.05)
    Em_list.extend([1] * first_5_percent)

    if isinstance(local_updates_mode, int):
        Em_list.extend([local_updates_mode] * (rounds - first_5_percent))
    elif isinstance(local_updates_mode, str) and local_updates_mode == 'log':
        Em_list.extend([int(math.ceil(math.log2(i + 1))) for i in range(1, rounds - first_5_percent+ 1)])
    
    n_samples = sum(Em_list) * n_clients
    p = 6  # dimension of covariates
    
    # Generate data
    np.random.seed(42)
    Sigma = np.array([[0.5 ** abs(j1 - j2) for j2 in range(p)] for j1 in range(p)])
    X_covariates = np.random.multivariate_normal(
        mean=np.zeros(p),
        cov=Sigma,
        size=n_samples
    )  # shape: (n_samples, p)
    
    epsilon = np.random.randn(n_samples)
    y = 1 + np.sum(X_covariates, axis=1) + epsilon

    # Convert to PyTorch tensors
    X = torch.from_numpy(X_covariates).float()
    y = torch.from_numpy(y).float()
    return X, y, Em_list


def load_data(context=Context):
    """load data for quantile regression."""

    global _global_data
    if _global_data is None:
        _global_data = generate_data(
            rounds=context.run_config['num-server-rounds'],
            local_updates_mode=context.run_config['local-updates-mode'], 
            n_clients=context.node_config['num-partitions'] 
        )

    X, y, Em_list = _global_data
    
    # get data for this partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    n_samples = len(y)
    per_partition_num = n_samples // num_partitions
    start_idx = partition_id * per_partition_num
    end_idx = (partition_id + 1) * per_partition_num

    # Create dataset and dataloaders
    dataset = TensorDataset(X[start_idx:end_idx], y[start_idx:end_idx])
    trainloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    return trainloader, Em_list


def lr_schedule(step,c0=0.01,a=0.51,b=0):
    """Learning rate schedule"""
    lr = c0 / (step**a + b)
    return lr

def train(net, tau, r, trainloader, Em_list, server_rounds_cnt, device):
    """Train the model with LDP mechanism."""
    net.to(device)
    tau_tilde = r * tau + (1 - r) / 2
    local_iter_nums = Em_list[server_rounds_cnt]
    begin_idx = sum(Em_list[:server_rounds_cnt])

    # compute lr for current round
    effective_lr =  lr_schedule(step=server_rounds_cnt+1)
    lr = effective_lr / local_iter_nums

    running_loss = 0.0

    # Perform specified number of SGD updates
    for _, (x, y) in enumerate(itertools.islice(trainloader, begin_idx, begin_idx + local_iter_nums)): 
        x, y = x.to(device), y.to(device)
        y_pred = net(x)
        z_true = (y <= y_pred).float()
        
        # Apply LDP perturbation
        if torch.rand(1).item() < r:
            z_tilde = z_true
        else:
            z_tilde = torch.rand(1).item() < 0.5
            z_tilde = torch.tensor(z_tilde, dtype=torch.float32, device=device)
        
        # Calculate gradients manually
        net.zero_grad()
        net.linear.weight.grad = x * (z_tilde - tau_tilde)  # torch.Size([1, 6])
        net.linear.bias.grad = (z_tilde - tau_tilde).view(-1)  # torch.Size([1])
        
        # Update parameters
        with torch.no_grad():
            net.linear.weight -= lr * net.linear.weight.grad
            net.linear.bias -= lr * net.linear.bias.grad
        
        # Calculate pinball loss for monitoring
        residuals = y - y_pred
        pinball_loss = torch.mean(torch.where(residuals >= 0,
                                            tau * residuals,
                                            (tau - 1) * residuals))
        running_loss += pinball_loss.item()
    
    avg_loss = running_loss / local_iter_nums
    return avg_loss


def get_weights(net):
    """Get model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)