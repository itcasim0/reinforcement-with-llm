"""
Return과 Advantage 추정 함수들

이 모듈은 강화학습에서 사용되는 다양한 Return과 Advantage 추정 방법을 제공합니다.
"""

import torch


def compute_mc(rewards, values, dones, gamma: float, device: torch.device):
    """
    Monte Carlo (MC) 방식으로 Return과 Advantage 계산

    알고리즘:
    - Return: G_t = r_t + γ * G_{t+1} (역순 계산)
    - Advantage: A_t = G_t - V(s_t)

    Args:
        rewards: 보상 리스트
        values: 가치 함수 예측값 리스트
        dones: 종료 플래그 리스트
        gamma: 할인율 (discount factor)
        device: 텐서를 생성할 디바이스

    Returns:
        returns: 할인된 누적 보상
        advantages: MC로 계산된 advantage
    """

    def to_scalar_float(x):
        """값을 스칼라 float로 변환"""
        if isinstance(x, torch.Tensor):
            return float(x.item())
        elif isinstance(x, (list, tuple)):
            return float(x[0]) if len(x) > 0 else 0.0
        else:
            return float(x)

    rewards = [to_scalar_float(r) for r in rewards]
    values = [to_scalar_float(v) for v in values]

    returns = []
    G = 0.0
    for r, done in zip(reversed(rewards), reversed(dones)):
        if done:
            G = 0.0
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    values_t = torch.tensor(values, dtype=torch.float32, device=device)
    advantages = returns - values_t

    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
    else:
        advantages = advantages - advantages.mean()

    return returns, advantages


def compute_gae(
    rewards, values, dones, gamma: float, gae_lambda: float, device: torch.device
):
    """
    GAE (Generalized Advantage Estimation) 방식으로 Return과 Advantage 계산

    알고리즘:
    - TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    - GAE: A_t = δ_t + (γλ) * A_{t+1} (역순 계산)
    - Return: R_t = A_t + V(s_t)

    수식:
    A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}

    파라미터:
    - λ = 0: TD(0)와 동일 (low variance, high bias)
    - λ = 1: Monte Carlo와 유사 (high variance, low bias)
    - λ ∈ (0,1): bias-variance 트레이드오프 조절

    Args:
        rewards: 보상 리스트
        values: 가치 함수 예측값 리스트
        dones: 종료 플래그 리스트
        gamma: 할인율 (discount factor)
        gae_lambda: GAE lambda 파라미터
        device: 텐서를 생성할 디바이스

    Returns:
        returns: 할인된 누적 보상
        advantages: GAE로 계산된 advantage
    """

    def to_scalar_float(x):
        """값을 스칼라 float로 변환"""
        if isinstance(x, torch.Tensor):
            return float(x.item())
        elif isinstance(x, (list, tuple)):
            return float(x[0]) if len(x) > 0 else 0.0
        else:
            return float(x)

    rewards = [to_scalar_float(r) for r in rewards]
    values = [to_scalar_float(v) for v in values]

    advantages = []
    returns = []

    gae = 0.0
    next_value = 0.0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0.0
            gae = 0.0

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae

        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])

        next_value = values[t]

    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
    else:
        advantages = advantages - advantages.mean()

    return returns, advantages
