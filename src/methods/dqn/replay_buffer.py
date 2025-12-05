"""
Experience Replay Buffer for DQN

DQN의 핵심 요소 중 하나인 Experience Replay를 구현합니다.
과거 경험을 저장하고 무작위로 샘플링하여 학습에 사용합니다.
"""

import random
from collections import deque
import torch


class ReplayBuffer:
    """
    Experience Replay Buffer
    
    DQN에서 사용하는 경험 저장소입니다.
    - (state, action, reward, next_state, done) 튜플을 저장
    - FIFO 방식으로 오래된 경험은 자동으로 제거
    - 무작위 샘플링으로 correlation 감소
    """

    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: 버퍼의 최대 크기
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        경험을 버퍼에 추가
        
        Args:
            state: 현재 상태 (torch.Tensor)
            action: 선택한 액션 (int)
            reward: 받은 보상 (float)
            next_state: 다음 상태 (torch.Tensor)
            done: 종료 플래그 (bool)
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        """
        버퍼에서 무작위로 배치 샘플링
        
        Args:
            batch_size: 샘플링할 배치 크기
            device: 텐서를 생성할 디바이스
            
        Returns:
            states: (batch_size, state_dim)
            actions: (batch_size,)
            rewards: (batch_size,)
            next_states: (batch_size, state_dim)
            dones: (batch_size,)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 텐서로 변환
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """버퍼의 현재 크기 반환"""
        return len(self.buffer)
