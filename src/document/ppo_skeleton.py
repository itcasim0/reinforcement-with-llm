import random
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


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
        logits = self.actor(x)               # (batch, num_actions)
        values = self.critic(x).squeeze(-1)  # (batch,)
        return logits, values


class PPORunner:
    """
    - 환경에서 rollout 수집
    - PPO loss 계산 및 업데이트

    관측 벡터(obs) 구성:
      [ g/5, r/5, c/5, o/5, step_norm, one_hot(last_action, num_actions) ]
    """

    def __init__(
        self,
        env,
        state_dim: int,       # = 4 + 1 + num_actions
        num_actions: int,
        gamma: float = 0.95,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        K_epochs: int = 4,
    ):
        self.env = env
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.K_epochs = K_epochs

        self.num_actions = num_actions
        self.state_dim = state_dim

        self.policy = PPOPolicy(state_dim=state_dim, num_actions=num_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    # -----------------------------
    # 관측 벡터 생성: (g,r,c,o) + step + last_action
    # -----------------------------
    def _build_obs(
        self,
        state: Tuple[float, float, float, float],  # ← float로
        step_idx: int,
        last_action_idx: int,
        max_steps: int,
    ) -> torch.Tensor:
        """
        state: (g, r, c, o) 정수
        step_idx: 0,1,... (현재 step 인덱스)
        last_action_idx: 직전 액션 인덱스, 없으면 -1
        """
        g, r, c, o = state
        base = torch.tensor([g, r, c, o], dtype=torch.float32) / 10.0

        # step_norm: 0 ~ 1
        step_norm = torch.tensor(
            [step_idx / max_steps],
            dtype=torch.float32,
        )

        # last_action one-hot
        last_onehot = torch.zeros(self.num_actions, dtype=torch.float32)
        if 0 <= last_action_idx < self.num_actions:
            last_onehot[last_action_idx] = 1.0

        obs = torch.cat([base, step_norm, last_onehot], dim=0)  # (4 + 1 + num_actions,)
        return obs

    # -----------------------------
    # trajectory 수집
    # -----------------------------
    def collect_trajectory(self, max_steps: int):
        """
        에피소드 하나를 rollout:
        - obs, states, actions, log_probs, rewards, dones, values 수집
        """
        obs_list = []
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        state, _ = self.env.reset()
        last_action_idx = -1  # 첫 step에는 이전 액션 없음

        for t in range(max_steps):
            obs = self._build_obs(
                state=state,
                step_idx=t,
                last_action_idx=last_action_idx,
                max_steps=max_steps,
            )
            obs_tensor = obs.unsqueeze(0)  # (1, state_dim)

            with torch.no_grad():
                logits, value = self.policy(obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            action_idx = int(action.item())
            next_state, reward, done, info = self.env.step(action_idx)

            # 버퍼에 저장
            obs_list.append(obs.detach().clone())
            states.append(state)
            actions.append(action_idx)
            log_probs.append(float(log_prob.item()))
            rewards.append(float(reward))
            dones.append(done)
            values.append(float(value.item()))

            # 다음 step 준비
            last_action_idx = action_idx
            state = next_state
            if done:
                break

        traj = {
            "obs": obs_list,       # 확장된 관측 벡터
            "states": states,      # (디버깅용) 원래 (g,r,c,o)
            "actions": actions,
            "log_probs": log_probs,
            "rewards": rewards,
            "dones": dones,
            "values": values,
        }
        return traj

    # -----------------------------
    # return & advantage 계산 (MC 버전)
    # -----------------------------
    def compute_gae(self, rewards, values, dones):
        """
        간단 버전: GAE 대신, MC return + advantage = R - V.
        """
        returns = []
        G = 0.0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0.0
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)
        advantages = returns - values_t

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        else:
            advantages = advantages - advantages.mean()

        return returns, advantages

    # -----------------------------
    # PPO 업데이트
    # -----------------------------
    def ppo_update(self, traj):
        obs_list = traj["obs"]
        actions = traj["actions"]
        old_log_probs = traj["log_probs"]
        rewards = traj["rewards"]
        dones = traj["dones"]
        values = traj["values"]

        obs_t = torch.stack(obs_list, dim=0)   # (T, state_dim)
        actions_t = torch.tensor(actions, dtype=torch.long)       # (T,)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32)  # (T,)

        returns, advantages = self.compute_gae(rewards, values, dones)

        for _ in range(self.K_epochs):
            logits, value_preds = self.policy(obs_t)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(value_preds, returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # -----------------------------
    # Train 루프
    # -----------------------------
    def train(self, num_episodes: int = 10, max_steps: int = 3):
        reward_history = []

        for ep in range(1, num_episodes + 1):
            traj = self.collect_trajectory(max_steps=max_steps)
            ep_return = sum(traj["rewards"])
            reward_history.append(ep_return)

            print(f"[Episode {ep}] return = {ep_return:.3f}, len = {len(traj['rewards'])}")

            self.ppo_update(traj)

        return reward_history

    # -----------------------------
    # Greedy 평가
    # -----------------------------
    def evaluate_greedy(self, max_steps: int = 3):
        """
        학습된 정책을 greedy로 평가:
        - 각 step에서 argmax(logits)로 행동 선택
        """
        state, text = self.env.reset()
        print("\n[Eval] 초기 문서:")
        print("-" * 60)
        print(text)
        print("-" * 60)
        print("초기 점수:", self.env.current_scores)

        actions_taken = []
        last_action_idx = -1

        for t in range(max_steps):
            obs = self._build_obs(
                state=state,
                step_idx=t,
                last_action_idx=last_action_idx,
                max_steps=max_steps,
            )
            obs_t = obs.unsqueeze(0)

            with torch.no_grad():
                logits, _ = self.policy(obs_t)
                action = torch.argmax(logits, dim=-1)

            action_idx = int(action.item())
            actions_taken.append(action_idx)
            action_name = self.env.actions[action_idx]

            before_text = self.env.current_text
            next_state, reward, done, info = self.env.step(action_idx)

            print(f"\n[Step {t+1}] action = {action_name}, reward = {reward:.3f}")
            print("  prev scores:", info.get("prev_scores"))
            print("  new  scores:", info.get("new_scores"))
            print("[before]")
            print(before_text)
            print("[after]")
            print(self.env.current_text)

            last_action_idx = action_idx
            state = next_state
            if done:
                print(f"\n[Eval] 종료 (reason={info.get('reason', 'unknown')}, step={t+1})")
                break

        print("\n[Eval] 최종 점수:", self.env.current_scores)
        print("선택된 액션 인덱스 시퀀스:", actions_taken)

if __name__ == "__main__":
    # 여기는 네 main.py에 있는 것들을 그대로 재사용한다고 가정
    from main import (
        get_openrouter_client,
        create_pararev_documents,
        OpenRouterEditorLLM,
        OpenRouterJudgeLLM,
        EditingEnv,
    )

    random.seed(42)
    torch.manual_seed(42)

    client = get_openrouter_client()
    # documents = create_coedit_documents(
    #     split="train",
    #     max_samples=50,       # LLM 호출 비용 감안해서 적당히
    #     task_filter=["gec"],  # 문법 교정 위주로 하고 싶다면
    # )

    documents = create_pararev_documents(max_docs=50)

    editor = OpenRouterEditorLLM(
        client=client,
        model="openai/gpt-4o-mini",
        base_cost=0.02,
    )
    judge = OpenRouterJudgeLLM(
        client=client,
        model="openai/gpt-4.1",
    )

    env = EditingEnv(
        documents=documents,
        editor=editor,
        judge=judge,
        max_steps=3,
        terminal_threshold=3.0,
        cost_lambda=1.0,
    )

    runner = PPORunner(
        env=env,
        state_dim=4 + 1 + env.num_actions,   # g,r,c,o + step + last_action_one_hot
        num_actions=env.num_actions,
        gamma=0.95,
        lr=3e-4,
        clip_eps=0.2,
        K_epochs=4,
    )

    print("=== PPO Skeleton: Train ===")
    """
    TODO: num_episodes 증가 (학습 데이터 수 더 많게)
    TODO: max_steps 늘렸을 때 stop action 을 빨리 선택하는 경우가 있는지 확인 
    """
    runner.train(num_episodes=30, max_steps=6)

    print("\n=== PPO Skeleton: Evaluate ===")
    runner.evaluate_greedy(max_steps=3)
