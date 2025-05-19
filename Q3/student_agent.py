import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Define log_std bounds for numerical stability
LOG_STD_MAX = 2
LOG_STD_MIN = -20 # Original SAC paper uses -20, some impl use -5 or -10
EPSILON = 1e-6 # For numerical stability in log_prob calculation

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class Actor(nn.Module):
    """
    Actor Network for SAC.
    Outputs parameters for a squashed Gaussian policy.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=512, action_bound=1.0):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.action_bound = torch.tensor(action_bound, dtype=torch.float32) # Store as tensor

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc = nn.Linear(hidden_dim, action_dim)
        self.log_std_fc = nn.Linear(hidden_dim, action_dim)

        # Ensure action_bound is on the same device as parameters later
        self.register_buffer('action_bound_const', self.action_bound)


    def forward(self, state):
        """
        Given a state, outputs mean and log_std for the action distribution.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_fc(x)
        log_std = self.log_std_fc(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

class Agent(object):
    def __init__(self):
        self.obs_dim = 67  
        self.act_dim = 21
        self.actor = Actor(self.obs_dim, self.act_dim).to(device)
        self.actor.load_state_dict(torch.load("./sac_actor_q3.pth", map_location=device))
        self.actor.eval()


    def act(self, observation):
        obs = torch.FloatTensor(observation).to(device)
        with torch.no_grad():
            mean, log_std = self.actor(obs) 
            action = torch.tanh(mean) 
        
        return action.detach().cpu().numpy() # Ensure correct dtype for submission