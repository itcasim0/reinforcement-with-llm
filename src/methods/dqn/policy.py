# external
import torch
import torch.nn as nn


class DQNPolicy(nn.Module):
    """
    DQN Q-Network
    
    확장된 상태 벡터(obs)를 입력으로 받아:
    - 각 액션에 대한 Q-value를 출력
    
    DQN은 value-based 방법으로, Q(s,a)를 학습하여 최적의 행동을 선택합니다.
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64):
        """
        Args:
            state_dim: 입력 상태 벡터의 차원
            num_actions: 가능한 액션의 개수
            hidden_dim: 은닉층의 차원 (기본값: 64)
        """
        super().__init__()
        
        # Q-network
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),  # 각 액션에 대한 Q-value 출력
        )

    def forward(self, obs: torch.Tensor):
        """
        순전파: 상태로부터 Q-value 계산
        
        Args:
            obs: (batch, state_dim) - 배치 단위의 상태 벡터
            
        Returns:
            q_values: (batch, num_actions) - 각 액션에 대한 Q-value
        """
        return self.network(obs)
