# external
import torch
import torch.nn as nn


class PPOPolicy(nn.Module):
    """확장된 상태 벡터(obs) -> 정책(액션 분포) + 가치 V(s)"""

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch, state_dim)
        """
        x = self.base(obs)
        logits = self.actor(x)  # (batch, num_actions)
        values = self.critic(x).squeeze(-1)  # (batch,)
        return logits, values
