import torch
import torch.nn as nn

import utils

class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()
        # TODO: Define the Q network
        self.Q = nn.Sequential(
            nn.Linear(repr_dim + action_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        # TODO: Define the forward pass 
        x = torch.cat([obs, action], dim=1)
        q = self.Q(x)
        return q
