
import torch
import torch.nn as nn
import utils

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, action_shape[0])
        self.log_std_layer = nn.Linear(hidden_dim, action_shape[0])
        self.apply(utils.weight_init)

    def forward(self, obs):
        x = self.trunk(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-10, max=2)
        return mu, log_std

    def get_dist(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        # e.g. use a truncated normal or normal distribution
        dist = utils.TruncatedNormal(mu, std)
        return dist
