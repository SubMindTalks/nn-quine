import numpy as np
import torch
from torch import nn

class RealQuine(nn.Module):
    def __init__(self, n_hidden, n_layers, act_func):
        super().__init__()

        # Create layers and parameter lists
        self.param_list = []
        self.param_names = []
        layers = []
        for i in range(n_layers):
            n_out = 1 if i == (n_layers - 1) else n_hidden
            current_layer = nn.Linear(n_hidden, n_out, bias=True)
            layers.append(current_layer)
            layers.append(act_func())

            self.param_list.append(current_layer.weight)
            self.param_names.append(f"layer{i + 1}_weight")
            self.param_list.append(current_layer.bias)
            self.param_names.append(f"layer{i + 1}_bias")
        layers.pop(-1)  # Remove final activation

        # Create the parameter counting function
        self.num_params_arr = np.array([np.prod(p.shape) for p in self.param_list])
        self.cum_params_arr = np.cumsum(self.num_params_arr)
        self.num_params = int(self.cum_params_arr[-1])

        # Main network
        self.net = nn.Sequential(*layers)

        # Self-replication output layer
        self.replication_layer = nn.Linear(n_hidden, self.num_params, bias=True)

    def forward(self, x, task="replication"):
        """
        Forward pass for both replication and task-solving.
        Args:
            x: Input data
            task: Either "replication" or "task"
        Returns:
            Output of the network for the given task
        """
        features = self.net(x)
        if task == "replication":
            return self.replication_layer(features)
        return features  # Default forward task

    def get_param(self, idx):
        """
        Retrieve a specific parameter for replication tasks.
        Args:
            idx: Index of the parameter
        Returns:
            The parameter value
        """
        assert idx < self.num_params
        subtract = 0
        param = None
        normalized_idx = None
        for i, n_params in enumerate(self.cum_params_arr):
            if idx < n_params:
                param = self.param_list[i]
                normalized_idx = idx - subtract
                break
            else:
                subtract = n_params
        return param.view(-1)[normalized_idx]
