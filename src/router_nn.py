"""
Router-style Multi-Action Environment + PPO Agent (Skeleton)

Actions:
  0: INSTANT_LLM
  1: THINK_LLM
  2: RAG
  3: TOOL
  4: STOP

이 코드는 구조를 보여주는 예시입니다.
- LLM / RAG / TOOL 호출 부분은 dummy로 되어 있고,
- 실제 연구에서는 여기를 LLM API, RAG 시스템, 도구 호출로 교체하면 됩니다.
"""

import random
import numpy as np
from enum import IntEnum
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


# =========================
# 1. 액션 정의
# =========================


class Actions(IntEnum):
    INSTANT_LLM = 0
    THINK_LLM = 1
    RAG = 2
    TOOL = 3
    STOP = 4


# =========================
# 2. 환경 정의 (RouterEnv)
# =========================


class RouterEnv:
    """
    간단한 시뮬레이션 환경:
      - state: [step_t, cost_so_far, num_instant, num_think, num_rag, num_tool]
      - action: INSTANT_LLM / THINK_LLM / RAG / TOOL / STOP
      - reward: (dummy) 성능 보상 + 비용 패널티
    실제 연구에서는:
      - call_* 함수에서 LLM / RAG / TOOL을 실제로 호출하고,
      - outcome_score를 실제 정답/품질 평가로 바꾸면 됩니다.
    """

    def __init__(
        self, max_steps: int = 5, alpha: float = 0.3, seed: int = 42  # cost trade-off
    ):
        self.max_steps = max_steps
        self.alpha = alpha
        self.rng = random.Random(seed)
        self.n_actions = len(Actions)
        self.state_dim = 6  # [step, cost, ns, nt, nr, ntl]

        self.reset()

    def reset(self):
        # 에피소드마다 "난이도"를 랜덤으로 부여 (0~1)
        # 난이도가 높을수록 THINK / RAG가 도움이 되는 식으로 시뮬레이트
        self.difficulty = self.rng.uniform(0.0, 1.0)

        self.step_t = 0
        self.cost_so_far = 0.0
        self.num_instant = 0
        self.num_think = 0
        self.num_rag = 0
        self.num_tool = 0

        self.done = False

        return self._get_state()

    def _get_state(self):
        # 간단하게 수치형 state만 사용
        return np.array(
            [
                self.step_t,
                self.cost_so_far,
                self.num_instant,
                self.num_think,
                self.num_rag,
                self.num_tool,
            ],
            dtype=np.float32,
        )

    # ---- 아래 call_* 함수들은 실제 연구에서 교체 포인트 ----

    def call_instant_llm(self):
        """
        dummy: 간단한 LLM 호출 비용/효과를 시뮬레이션
        - cost: 작음
        - 정보 획득 정도: 난이도가 낮을수록 잘 맞춘다고 가정
        """
        cost = self.rng.uniform(1.0, 3.0)  # 토큰 비용 등
        # 난이도가 낮을수록 instant가 잘 먹힌다고 가정
        info_gain = max(0.0, 0.6 - self.difficulty + self.rng.uniform(-0.1, 0.1))
        return cost, info_gain

    def call_think_llm(self):
        """
        dummy: 깊이 생각하는 LLM
        - cost: 큼
        - 정보 획득 정도: 난이도 높을수록 이쪽이 유리
        """
        cost = self.rng.uniform(4.0, 8.0)
        info_gain = max(0.0, 0.2 + self.difficulty + self.rng.uniform(-0.1, 0.1))
        return cost, info_gain

    def call_rag(self):
        """
        dummy: RAG 호출
        - cost: 중간
        - 정보 획득 정도: 난이도와 상관 없이 중간 이상이라고 가정
        """
        cost = self.rng.uniform(3.0, 6.0)
        info_gain = max(0.0, 0.4 + self.rng.uniform(-0.1, 0.1))
        return cost, info_gain

    def call_tool(self):
        """
        dummy: Tool (계산기 등)
        - cost: 작음
        - 정보 획득: 특정 난이도 구간(예: 수학 문제)에만 도움이 된다고 가정 가능
        여기서는 난이도와 상관 없는 작은 정보 획득으로 둠.
        """
        cost = self.rng.uniform(0.5, 1.5)
        info_gain = max(0.0, 0.2 + self.rng.uniform(-0.1, 0.1))
        return cost, info_gain

    # --------------------------------------------------------

    def _compute_episode_outcome(self):
        """
        dummy: 지금까지의 info_gain과 difficulty를 바탕으로 '정답률' 비슷한 스코어 산출.
        실제로는:
          - LLM이 생성한 최종 답변을 평가(EM/F1/루브릭 등)
        로 교체해야 함.
        """
        # 간단히: THINK/RAG 사용량이 많을수록,
        # 그리고 난이도가 높을수록 더 많이 써야 좋은 성능이 나온다고 가정
        base_score = 0.3 + 0.1 * self.num_instant
        base_score += 0.15 * self.num_think + 0.15 * self.num_rag
        base_score += 0.05 * self.num_tool

        # 난이도 높은 경우에는 THINK/RAG가 없으면 penalty
        if self.difficulty > 0.7 and (self.num_think + self.num_rag) == 0:
            base_score -= 0.4

        # 0~1로 클리핑
        return float(np.clip(base_score, 0.0, 1.0))

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        reward = 0.0
        info = {}

        if action == Actions.INSTANT_LLM:
            cost, info_gain = self.call_instant_llm()
            self.cost_so_far += cost
            self.num_instant += 1

        elif action == Actions.THINK_LLM:
            cost, info_gain = self.call_think_llm()
            self.cost_so_far += cost
            self.num_think += 1

        elif action == Actions.RAG:
            cost, info_gain = self.call_rag()
            self.cost_so_far += cost
            self.num_rag += 1

        elif action == Actions.TOOL:
            cost, info_gain = self.call_tool()
            self.cost_so_far += cost
            self.num_tool += 1

        elif action == Actions.STOP:
            # 에피소드 종료 → 최종 보상 계산
            outcome_score = self._compute_episode_outcome()

            # 성능 보상(0~1) - 비용 패널티(정규화 없이 간단)
            # 실제로는 비용을 0~1로 노멀라이즈해서 쓰는 게 안정적
            perf_reward = outcome_score
            cost_penalty = self.alpha * self.cost_so_far
            reward = perf_reward - cost_penalty

            self.done = True
            self.step_t += 1
            return self._get_state(), reward, self.done, info

        else:
            raise ValueError(f"Unknown action: {action}")

        # STOP을 선택하지 않은 경우: 중간 timestep
        self.step_t += 1
        if self.step_t >= self.max_steps:
            # 강제 종료 + 보상 계산
            outcome_score = self._compute_episode_outcome()
            perf_reward = outcome_score
            cost_penalty = self.alpha * self.cost_so_far
            reward = perf_reward - cost_penalty
            self.done = True

        # 중간 step에는 보상을 0으로 줄 수도 있고,
        # 여기처럼 에피소드 전체 보상을 한 번에 주는 구조도 가능.
        # 지금은 "STOP 또는 max_steps"에서만 보상이 터지게 설정함.
        if not self.done:
            reward = 0.0

        return self._get_state(), reward, self.done, info


# =========================
# 3. PPO 구성 요소
# =========================


class ActorCritic(nn.Module):
    """
    Actor-Critic 네트워크
    - Actor: 정책 π(a|s)를 출력 (action probabilities)
    - Critic: 가치 함수 V(s)를 출력
    """

    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super().__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, n_actions)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor output: action probabilities
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic output: state value
        state_value = self.critic(x)
        
        return action_probs, state_value

    def get_action(self, state, deterministic=False):
        """
        상태가 주어졌을 때 액션을 샘플링
        """
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        
        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        action_log_prob = dist.log_prob(action)
        
        return action, action_log_prob, state_value

    def evaluate_actions(self, states, actions):
        """
        주어진 상태-액션 쌍에 대한 log_prob, value, entropy 계산
        """
        action_probs, state_values = self.forward(states)
        dist = Categorical(action_probs)
        
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return action_log_probs, state_values, entropy


class RolloutBuffer:
    """
    PPO를 위한 경험 저장 버퍼
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def push(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def get(self):
        return (
            self.states,
            self.actions,
            self.rewards,
            self.log_probs,
            self.values,
            self.dones,
        )

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) Agent
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.device = device

        # Actor-Critic 네트워크
        self.policy = ActorCritic(state_dim, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Rollout buffer
        self.buffer = RolloutBuffer()

    def select_action(self, state, deterministic=False):
        """
        액션 선택 (학습 중에는 stochastic, 평가 시에는 deterministic)
        """
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_t, deterministic)
        
        return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones):
        """
        Generalized Advantage Estimation (GAE) 계산
        """
        advantages = []
        gae = 0
        
        # 역순으로 계산
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE: A_t = δ_t + γλ * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages

    def update(self):
        """
        PPO 업데이트
        """
        if len(self.buffer) == 0:
            return

        # 버퍼에서 데이터 가져오기
        states, actions, rewards, old_log_probs, values, dones = self.buffer.get()

        # Numpy array로 변환
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        old_log_probs = np.array(old_log_probs)
        values = np.array(values)
        dones = np.array(dones)

        # GAE 계산
        advantages = self.compute_gae(rewards, values, dones)
        advantages = np.array(advantages)
        
        # Returns 계산: R_t = A_t + V(s_t)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Tensor로 변환
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(
            old_log_probs, dtype=torch.float32, device=self.device
        )
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # PPO epochs
        for _ in range(self.ppo_epochs):
            # 현재 정책으로 log_probs, values, entropy 계산
            log_probs, values, entropy = self.policy.evaluate_actions(
                states_t, actions_t
            )

            # Ratio: π_new / π_old
            ratio = torch.exp(log_probs - old_log_probs_t)

            # Surrogate loss
            surr1 = ratio * advantages_t
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * advantages_t
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values = values.squeeze()
            value_loss = F.mse_loss(values, returns_t)

            # Entropy bonus (exploration)
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                actor_loss
                + self.value_coef * value_loss
                + self.entropy_coef * entropy_loss
            )

            # Gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # 버퍼 클리어
        self.buffer.clear()


def train_ppo(
    num_episodes: int = 2000,
    max_steps: int = 5,
    alpha: float = 0.3,
    update_interval: int = 10,  # 몇 에피소드마다 업데이트할지
    device: str = "cpu",
):
    env = RouterEnv(max_steps=max_steps, alpha=alpha)
    agent = PPOAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        gamma=0.99,
        gae_lambda=0.95,
        lr=3e-4,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        device=device,
    )

    episode_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0

        for t in range(max_steps + 1):  # STOP 포함
            # 액션 선택
            action, log_prob, value = agent.select_action(state, deterministic=False)
            
            # 환경 step
            next_state, reward, done, info = env.step(action)

            # 버퍼에 저장
            agent.buffer.push(state, action, reward, log_prob, value, done)

            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

        # 일정 에피소드마다 업데이트
        if (ep + 1) % update_interval == 0:
            agent.update()

        # 로그 출력
        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(
                f"[Episode {ep+1:4d}] avg_reward(last 100) = {avg_reward:.4f}, "
                f"episode_reward = {episode_reward:.4f}"
            )

    return agent


if __name__ == "__main__":
    # CPU에서 간단히 돌려볼 수 있는 구조
    # 실제 연구에서는 device="cuda"로 바꿔서 사용
    trained_agent = train_ppo(
        num_episodes=2000,
        max_steps=5,
        alpha=0.1,
        update_interval=10,
        device="cpu",
    )
