import torch
from torch import nn
import numpy as np
from sklearn import random_projection

class RealQuine(nn.Module):
    """
    RealQuine extends VanillaQuine to include additional tasks or mechanisms
    beyond simple weight replication.
    """
    def __init__(self, n_hidden, n_layers, act_func):
        super().__init__()

        # Initialize parameters and layers
        self.param_list = []
        self.param_names = []
        layers = []

        # Create model layers
        for i in range(n_layers):
            n_out = 1 if i == (n_layers - 1) else n_hidden
            current_layer = nn.Linear(n_hidden, n_out, bias=True)
            layers.append(current_layer)
            layers.append(act_func())

            # Track weights and biases for replication
            self.param_list.append(current_layer.weight)
            self.param_names.append(f"layer{i + 1}_weight")
            self.param_list.append(current_layer.bias)
            self.param_names.append(f"layer{i + 1}_bias")

        # Remove final activation (not needed for output layer)
        layers.pop(-1)

        # Parameter metadata for quine
        self.num_params_arr = np.array([np.prod(p.shape) for p in self.param_list])
        self.cum_params_arr = np.cumsum(self.num_params_arr)
        self.num_params = int(self.cum_params_arr[-1])

        # Random projection matrix for parameter encoding
        X = np.random.rand(1, self.num_params)
        transformer = random_projection.GaussianRandomProjection(n_components=n_hidden)
        transformer.fit(X)
        rand_proj_matrix = transformer.components_

        # Random projection layer (non-trainable)
        rand_proj_layer = nn.Linear(self.num_params, n_hidden, bias=False)
        rand_proj_layer.weight.data = torch.tensor(rand_proj_matrix, dtype=torch.float32)
        for p in rand_proj_layer.parameters():
            p.requires_grad_(False)
        layers.insert(0, rand_proj_layer)

        # Final network
        self.net = nn.Sequential(*layers)

    def forward(self, x, task="replication"):
        """
        Forward pass for the RealQuine.

        Args:
            x (torch.Tensor): Input tensor.
            task (str): Task type. Default is "replication".
        """
        if task == "replication":
            return self.net(x)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def get_param(self, idx):
        """
        Retrieve the parameter at the specified index.

        Args:
            idx (int): Index of the parameter.

        Returns:
            torch.Tensor: The parameter at the given index.
        """
        assert idx < self.num_params, "Parameter index out of bounds."
        subtract = 0
        param = None

        for i, n_params in enumerate(self.cum_params_arr):
            if idx < n_params:
                param = self.param_list[i]
                normalized_idx = idx - subtract
                break
            else:
                subtract = n_params

        return param.view(-1)[normalized_idx]
