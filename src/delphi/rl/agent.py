import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent:
    def __init__(self,envs, agent_id) -> None:
        self.agent_id = agent_id
        self.policy = Policy(envs, agent_id)
        self.policy.critic = self.policy.critic.to(envs.device)
        self.policy.actor_mean = self.policy.actor_mean.to(envs.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=envs.learning_rate, eps=1e-5)
        
class Policy(nn.Module):
    def __init__(self, envs, agent_id):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_spaces[agent_id].shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_spaces[agent_id].shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, np.array(envs.action_spaces[agent_id].shape).prod()), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(np.prod(envs.action_spaces[agent_id].shape)))
        self.dev = envs.device

    def get_value(self, x):
        x = x.to(self.dev)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = (x - x.min()) / (x.max() - x.min())
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd).to(action_mean.device)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(), probs.entropy().sum(), self.critic(x)