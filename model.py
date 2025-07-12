# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[256, 256]):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))  # 학습 가능한 log std

    def forward(self, obs):
        raise NotImplementedError("단일 forward는 사용 안함")

    def act(self, obs):
        mu  = self.actor(obs)                     # ★ tanh 제거
        std = self.log_std.exp()
        unsquashed = torch.distributions.Normal(mu, std).rsample()
        action     = torch.tanh(unsquashed)       # 단 한 번만 squash
        log_prob   = (-0.5 * ((unsquashed - mu) / std).pow(2)       # Normal log-pdf
                    - torch.log(std) - 0.5 * np.log(2*np.pi)).sum(-1)
        log_prob  -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)    # tanh 보정
        scale = torch.tensor([1.5, 4.0], device=action.device)
        return action * scale, log_prob, None     # dist 반환 안 써서 None

    def evaluate(self, obs, scaled_action):
        scale = torch.tensor([1.5, 4.0], device=obs.device)
        # unsquashed = torch.atanh(scaled_action / scale.clamp_min(1e-6))  # ★
        x = scaled_action / scale.clamp_min(1e-6)
        x = torch.clamp(x, -1 + 1e-6, 1 - 1e-6)     # <- 추가
        unsquashed = torch.atanh(x)
        mu  = self.actor(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        log_prob = dist.log_prob(unsquashed).sum(-1)
        log_prob -= torch.log(1 - torch.tanh(unsquashed).pow(2) + 1e-6).sum(-1)
        entropy  = dist.entropy().sum(-1)
        value    = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value
