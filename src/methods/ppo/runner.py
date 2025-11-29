from typing import Tuple
from pathlib import Path
import json

# external
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# internal
from environments.editing_env.base_env import EditingEnv
from environments.editing_env.offline_env import OfflineEditingEnv

from utils.util import today_datetime
from utils.logger_factory import log

from .policy import PPOPolicy

class PPORunner:
    """
    - 환경에서 rollout 수집
    - PPO loss 계산 및 업데이트

    관측 벡터(obs) 구성:
      [ g/5, r/5, c/5, o/5, step_norm, one_hot(last_action, num_actions) ]

    Args:
        env (EditingEnv): 강화학습 환경 (문서 편집 환경)
        max_steps (int): 에피소드당 최대 스텝 수
        state_dim (int): 상태 벡터의 차원 (= 4(점수) + 1(step_norm) + num_actions(이전 액션 one-hot))
        num_actions (int): 가능한 액션의 개수 (편집 작업 종류)
        gamma (float): 할인율 (discount factor) - 미래 보상의 가중치. 기본값 0.95
        lr (float): 학습률 (learning rate) - 옵티마이저의 스텝 크기. 기본값 3e-4
        clip_eps (float): PPO 클리핑 범위 - 정책 업데이트의 안정성 제어. 기본값 0.2
        K_epochs (int): PPO 업데이트 반복 횟수 - 수집된 데이터로 몇 번 학습할지. 기본값 3
    """

    def __init__(
        self,
        env: EditingEnv | OfflineEditingEnv,
        max_steps: 3,
        state_dim: int,  # = 4 + 1 + num_actions
        num_actions: int,
        gamma: float = 0.95,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        K_epochs: int = 3,
    ):
        # 디바이스 설정 (CUDA 사용 가능하면 CUDA, 아니면 CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"디바이스 설정: {self.device}")

        # 환경 초기화
        self.env: EditingEnv | OfflineEditingEnv = env
        self.max_steps = max_steps

        # PPOPolicy(신경망 모델) 설계
        self.num_actions = num_actions
        self.state_dim = state_dim

        # PPOPolicy(신경망 모델) 하이퍼파라미터
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.K_epochs = K_epochs

        self.policy = PPOPolicy(state_dim=state_dim, num_actions=num_actions).to(
            self.device
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 체크포인트 관련
        self.start_episode = 0  # 학습 시작 에피소드 번호
        self.checkpoint_session_dir = None  # 현재 학습 세션의 체크포인트 디렉토리
        self.best_score = -float("inf")  # 최고 점수 기록

    # -----------------------------
    # 관측 벡터 생성: (structure, length, academic_style, information_density, clarity, overall) + step + last_action
    # -----------------------------
    def _build_obs(
        self,
        state: Tuple[float, float, float, float, float, float],  # 6개 평가 기준
        step_idx: int,
        last_action_idx: int,
    ) -> torch.Tensor:
        """
        state: (structure, length, academic_style, information_density, clarity, overall)
        step_idx: 0,1,... (현재 step 인덱스)
        last_action_idx: 직전 액션 인덱스, 없으면 -1
        """
        structure, length, academic_style, information_density, clarity, overall = state
        base = (
            torch.tensor(
                [
                    structure,
                    length,
                    academic_style,
                    information_density,
                    clarity,
                    overall,
                ],
                dtype=torch.float32,
                device=self.device,
            )
            / 10.0
        )

        # step_norm: 0 ~ 1
        step_norm = torch.tensor(
            [step_idx / self.max_steps],
            dtype=torch.float32,
            device=self.device,
        )

        # last_action one-hot
        last_onehot = torch.zeros(
            self.num_actions, dtype=torch.float32, device=self.device
        )
        if 0 <= last_action_idx < self.num_actions:
            last_onehot[last_action_idx] = 1.0

        obs = torch.cat([base, step_norm, last_onehot], dim=0)  # (4 + 1 + num_actions,)
        return obs

    # -----------------------------
    # trajectory 수집
    # -----------------------------
    # TODO: trajectory를 1번(episode 1개) 수집하고 바로 학습하는 상태이므로, 여러 번 수집 후 학습하는 방향도 검토할 것
    def _collect_trajectory(self):
        """
        에피소드 하나를 rollout:
        - obs, states, actions, log_probs, rewards, dones, values, infos 수집
        """
        obs_list = []
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        infos = []  # 각 스텝의 info 저장

        state, _ = self.env.reset()
        last_action_idx = -1  # 첫 step에는 이전 액션 없음

        for t in range(self.max_steps):
            obs = self._build_obs(
                state=state, step_idx=t, last_action_idx=last_action_idx
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
            infos.append(info)  # info 저장

            # 다음 step 준비
            last_action_idx = action_idx
            state = next_state
            if done:
                break

        traj = {
            "obs": obs_list,  # 확장된 관측 벡터
            "states": states,  # (디버깅용) 원래 (g,r,c,o)
            "actions": actions,
            "log_probs": log_probs,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "infos": infos,  # info 추가
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
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)
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

        obs_t = torch.stack(obs_list, dim=0)  # (T, state_dim)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)  # (T,)
        old_log_probs_t = torch.tensor(
            old_log_probs, dtype=torch.float32, device=self.device
        )  # (T,)

        returns, advantages = self.compute_gae(rewards, values, dones)

        for _ in range(self.K_epochs):
            logits, value_preds = self.policy(obs_t)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(value_preds, returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # -----------------------------
    # 체크포인트 저장/로드
    # -----------------------------
    def save_checkpoint(self, checkpoint_dir: str, episode: int, is_best: bool = False):
        """
        체크포인트 저장

        Args:
            checkpoint_dir: 체크포인트를 저장할 디렉토리
            episode: 현재 에피소드 번호
            is_best: 현재 체크포인트가 최고 성능인지 여부
        """
        # 세션 디렉토리가 없으면 생성하지 않음 (train 메서드에서 생성)
        if self.checkpoint_session_dir is None:
            log.warning("체크포인트 세션 디렉토리가 설정되지 않았습니다.")
            return

        checkpoint_file = self.checkpoint_session_dir / f"checkpoint_ep{episode}.pt"

        checkpoint = {
            "episode": episode,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "gamma": self.gamma,
            "clip_eps": self.clip_eps,
            "K_epochs": self.K_epochs,
            "best_score": self.best_score,
        }

        torch.save(checkpoint, checkpoint_file)

        # 최신 체크포인트 파일 경로도 저장
        latest_file = self.checkpoint_session_dir / "latest_checkpoint.txt"
        with open(latest_file, "w") as f:
            f.write(str(checkpoint_file))

        # 최고 성능 체크포인트 파일 경로 저장
        if is_best:
            best_file = self.checkpoint_session_dir / "best_checkpoint.txt"
            with open(best_file, "w") as f:
                f.write(str(checkpoint_file))

    def load_checkpoint(self, checkpoint_path: str, load_best: bool = False):
        """
        체크포인트 로드

        Args:
            checkpoint_path: 체크포인트 파일 경로 또는 디렉토리 경로
            load_best: True이면 best_checkpoint.txt 참조, False이면 latest_checkpoint.txt 참조

        Returns:
            로드된 에피소드 번호
        """
        path = Path(checkpoint_path)

        # 디렉토리가 주어진 경우
        if path.is_dir():
            if load_best:
                ptr_file = path / "best_checkpoint.txt"
                file_type = "best"
            else:
                ptr_file = path / "latest_checkpoint.txt"
                file_type = "latest"

            if ptr_file.exists():
                with open(ptr_file, "r") as f:
                    checkpoint_file = Path(f.read().strip())
            else:
                if load_best:
                    raise FileNotFoundError(
                        f"Best 체크포인트 참조 파일을 찾을 수 없습니다: {ptr_file}"
                    )

                # latest_checkpoint.txt가 없으면 가장 최근 파일 찾기
                checkpoints = sorted(path.glob("checkpoint_ep*.pt"))
                if not checkpoints:
                    raise FileNotFoundError(
                        f"체크포인트 파일을 찾을 수 없습니다: {path}"
                    )
                checkpoint_file = checkpoints[-1]
        else:
            checkpoint_file = path

        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"체크포인트 파일이 존재하지 않습니다: {checkpoint_file}"
            )

        log.info(f"체크포인트 로드: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        # 모델 및 옵티마이저 상태 복원
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_episode = checkpoint["episode"]

        # 최고 점수 복원
        if "best_score" in checkpoint:
            self.best_score = checkpoint["best_score"]

        # 체크포인트 세션 디렉토리 설정 (로드한 체크포인트의 부모 디렉토리)
        self.checkpoint_session_dir = checkpoint_file.parent
        log.info(f"체크포인트 세션 디렉토리 설정: {self.checkpoint_session_dir}")

        log.info(
            f"에피소드 {self.start_episode}부터 재개합니다. (best_score={self.best_score})"
        )

        return self.start_episode

    # -----------------------------
    # Trajectory 정보 저장
    # -----------------------------
    def save_trajectory_info(self, checkpoint_dir: str, episode: int, traj: dict):
        """
        에피소드의 trajectory 정보를 JSON 파일로 저장

        Args:
            checkpoint_dir: 체크포인트 디렉토리
            episode: 현재 에피소드 번호
            traj: trajectory 딕셔너리 (states, actions, rewards, infos 등)
        """
        # 세션 디렉토리가 없으면 생성하지 않음 (train 메서드에서 생성)
        if self.checkpoint_session_dir is None:
            log.warning("체크포인트 세션 디렉토리가 설정되지 않았습니다.")
            return

        # trajectory 정보를 저장할 디렉토리
        traj_dir = self.checkpoint_session_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)

        # 에피소드별 trajectory 파일
        traj_file = traj_dir / f"episode_{episode}.json"

        # 저장할 데이터 구성
        traj_data = {
            "episode": episode,
            "total_return": sum(traj["rewards"]),
            "num_steps": len(traj["rewards"]),
            "steps": [],
        }

        # 각 스텝별 정보 저장
        for step_idx in range(len(traj["rewards"])):
            # info 딕셔너리를 JSON 직렬화 가능하도록 변환
            info = traj["infos"][step_idx] if step_idx < len(traj["infos"]) else {}
            serializable_info = {}

            for key, value in info.items():
                if hasattr(value, "__dict__"):
                    # DocumentScore 같은 객체는 딕셔너리로 변환
                    serializable_info[key] = {
                        "structure": float(value.structure),
                        "length": float(value.length),
                        "academic_style": float(value.academic_style),
                        "information_density": float(value.information_density),
                        "clarity": float(value.clarity),
                        "overall": float(value.overall),
                    }
                else:
                    serializable_info[key] = value

            step_info = {
                "step": step_idx,
                "state": {
                    "structure": float(traj["states"][step_idx][0]),
                    "length": float(traj["states"][step_idx][1]),
                    "academic_style": float(traj["states"][step_idx][2]),
                    "information_density": float(traj["states"][step_idx][3]),
                    "clarity": float(traj["states"][step_idx][4]),
                    "overall": float(traj["states"][step_idx][5]),
                },
                "action": int(traj["actions"][step_idx]),
                "action_name": self.env.actions[traj["actions"][step_idx]],
                "reward": float(traj["rewards"][step_idx]),
                "log_prob": float(traj["log_probs"][step_idx]),
                "value": float(traj["values"][step_idx]),
                "done": bool(traj["dones"][step_idx]),
                "info": serializable_info,
            }
            traj_data["steps"].append(step_info)

        # JSON 파일로 저장
        with open(traj_file, "w", encoding="utf-8") as f:
            json.dump(traj_data, f, ensure_ascii=False, indent=2)

    # -----------------------------
    # Train 루프
    # -----------------------------
    def train(
        self,
        num_episodes: int = 10,
        checkpoint_dir: str = None,
        checkpoint_interval: int = 5,
        log_interval: int = 1,
        trajectory_save_interval: int = 1,
    ):
        """
        PPO 학습 루프

        Args:
            num_episodes: 총 학습할 에피소드 수
            checkpoint_dir: 체크포인트를 저장할 디렉토리 (None이면 저장 안 함)
            checkpoint_interval: 체크포인트 저장 주기 (에피소드 단위)
            log_interval: 로그 출력 주기 (에피소드 단위)
            trajectory_save_interval: Trajectory 정보 저장 주기 (에피소드 단위)
        """
        # 체크포인트 세션 디렉토리 초기화 (학습 시작 시 한 번만)
        if checkpoint_dir and self.checkpoint_session_dir is None:
            self.checkpoint_session_dir = Path(checkpoint_dir) / f"{today_datetime()}"
            self.checkpoint_session_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"체크포인트 세션 디렉토리 생성: {self.checkpoint_session_dir}")

        reward_history = []

        for ep in range(self.start_episode + 1, self.start_episode + num_episodes + 1):
            traj = self._collect_trajectory()
            ep_return = sum(traj["rewards"])
            reward_history.append(ep_return)

            if ep % log_interval == 0:
                log.info(
                    f"[Episode {ep}] return = {ep_return:.3f}, len = {len(traj['rewards'])}"
                )

            # 최고 점수 갱신 및 저장
            is_best = False
            if ep_return > self.best_score:
                self.best_score = ep_return
                is_best = True
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir, ep, is_best=True)

            self.ppo_update(traj)

            # trajectory 정보 저장
            if checkpoint_dir and ep % trajectory_save_interval == 0:
                self.save_trajectory_info(checkpoint_dir, ep, traj)

            # 체크포인트 저장 (주기적)
            if checkpoint_dir and ep % checkpoint_interval == 0:
                # 이미 best로 저장했으면 중복 저장 방지 (선택사항이지만 파일IO 줄이기 위해)
                if not is_best:
                    self.save_checkpoint(checkpoint_dir, ep)

        # 학습 종료 시 최종 체크포인트 저장
        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir, self.start_episode + num_episodes)

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
        log.info("\n[Eval] 초기 문서:")
        log.info("-" * 60)
        log.info(text)
        log.info("-" * 60)
        log.info(f"초기 점수: {self.env.current_score}")

        actions_taken = []
        last_action_idx = -1

        for t in range(max_steps):
            obs = self._build_obs(
                state=state,
                step_idx=t,
                last_action_idx=last_action_idx,
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

            log.info(f"\n[Step {t+1}] action = {action_name}, reward = {reward:.3f}")
            log.info(f"  prev scores: {info.get("prev_scores")}")
            log.info(f"  new  scores: {info.get("new_scores")}")
            log.info("[before]")
            log.info(before_text)
            log.info("[after]")
            log.info(self.env.current_text)

            last_action_idx = action_idx
            state = next_state
            if done:
                log.info(
                    f"\n[Eval] 종료 (reason={info.get('reason', 'unknown')}, step={t+1})"
                )
                break

        log.info(f"\n[Eval] 최종 점수: {self.env.current_score}")
        log.info(f"선택된 액션 인덱스 시퀀스: {actions_taken}")

    # -----------------------------
    # 정책 시각화
    # -----------------------------
    def show_policy(self):
        """학습된 정책의 액션 확률 분포 확인"""
        state, _ = self.env.reset()

        log.info(f"\n현재 문서 점수:")
        log.info(f"  structure:           {state[0]:.2f}")
        log.info(f"  length:              {state[1]:.2f}")
        log.info(f"  academic_style:      {state[2]:.2f}")
        log.info(f"  information_density: {state[3]:.2f}")
        log.info(f"  clarity:             {state[4]:.2f}")
        log.info(f"  overall:             {state[5]:.2f}")

        obs = self._build_obs(state, 0, -1)
        with torch.no_grad():
            logits, value = self.policy(obs.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        log.info(f"\nStep 1 액션 확률 (Value: {value.item():.3f}):")
        for action, prob in zip(self.env.actions, probs):
            bar = "*" * int(prob * 25)
            log.info(f"  {action:20s}: {prob:.3f} {bar}")
