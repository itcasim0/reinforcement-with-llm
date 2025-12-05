# external
import torch
import torch.nn as nn


class A2CPolicy(nn.Module):
    """
    A2C Actor-Critic 네트워크
    
    확장된 상태 벡터(obs)를 입력으로 받아:
    - Actor: 각 액션에 대한 로짓(정책) 출력
    - Critic: 현재 상태의 가치 V(s) 출력
    
    A2C 알고리즘에서 정책 개선과 가치 함수 학습을 동시에 수행하기 위한 공유 기반 구조
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64):
        """
        Args:
            state_dim: 입력 상태 벡터의 차원
            num_actions: 가능한 액션의 개수
            hidden_dim: 은닉층의 차원 (기본값: 64)
        """
        super().__init__()
        
        # 공유 특징 추출 네트워크 (Shared Base Network)
        # Actor와 Critic이 공유하는 feature extractor로, 상태를 압축된 표현으로 변환
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # 입력층
            nn.Tanh(),  # 비선형 활성화 함수
            nn.Linear(hidden_dim, hidden_dim),  # 은닉층
            nn.Tanh(),  # 비선형 활성화 함수
        )
        
        # Actor Head: 정책 네트워크 (각 액션에 대한 선호도 출력)
        self.actor = nn.Linear(hidden_dim, num_actions)
        
        # Critic Head: 가치 네트워크 (현재 상태의 가치 V(s) 추정)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        """
        순전파: 상태로부터 정책과 가치를 계산
        
        Args:
            obs: (batch, state_dim) - 배치 단위의 상태 벡터
            
        Returns:
            logits: (batch, num_actions) - 각 액션에 대한 로짓 (정책)
            values: (batch,) - 각 상태의 가치 추정값 V(s)
        """
        # 공유 기반 네트워크를 통한 특징 추출
        x = self.base(obs)
        
        # Actor: 액션 로짓 계산 (소프트맥스 전 값)
        logits = self.actor(x)  # (batch, num_actions)
        
        # Critic: 상태 가치 추정
        values = self.critic(x).squeeze(-1)  # (batch,) - 마지막 차원 제거
        
        return logits, values
