# utils.py

import torch

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    values = values + [last_value]                 # ★ bootstrap
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * (1 - dones[i]) - values[i]
        gae   = delta + gamma * lam * (1 - dones[i]) * gae
        advantages[i] = gae
    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    return torch.tensor(advantages, dtype=torch.float32), \
           torch.tensor(returns,     dtype=torch.float32)

