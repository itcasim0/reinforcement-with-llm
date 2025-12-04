from typing import Tuple
from pathlib import Path
import json
from dataclasses import fields
import time

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
from .estimators import compute_gae


class PPORunner:
    """
    - 환경에서 rollout 수집
    - PPO loss 계산 및 업데이트

    Args:
        env (EditingEnv): 강화학습 환경 (문서 편집 환경)
        max_steps (int): 에피소드당 최대 스텝 수
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
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        K_epochs: int = 3,
        buffer_size: int = 256,
        batch_size: int = 32,
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
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.K_epochs = K_epochs
        self.buffer_size = buffer_size  # 학습 전에 모을 step 수
        self.batch_size = batch_size  # 미니배치 크기

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
    # 여러 trajectory 합치기
    # -----------------------------
    def _merge_trajectories(self, trajs: list) -> dict:
        """
        여러 trajectory를 하나로 합침

        Args:
            trajs: trajectory 딕셔너리 리스트

        Returns:
            합쳐진 trajectory 딕셔너리
        """
        merged = {
            "obs": [],
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "infos": [],
        }

        for traj in trajs:
            merged["obs"].extend(traj["obs"])
            merged["states"].extend(traj["states"])
            merged["actions"].extend(traj["actions"])
            merged["log_probs"].extend(traj["log_probs"])
            merged["rewards"].extend(traj["rewards"])
            merged["dones"].extend(traj["dones"])
            merged["values"].extend(traj["values"])
            merged["infos"].extend(traj["infos"])

        return merged

    # -----------------------------
    # PPO 업데이트
    # -----------------------------
    def ppo_update(self, trajs):
        """
        여러 trajectory를 사용하여 PPO 업데이트 수행 (미니배치 학습)

        Args:
            trajs: trajectory 딕셔너리 또는 trajectory 딕셔너리 리스트
        """
        # 단일 trajectory인 경우 리스트로 변환
        if isinstance(trajs, dict):
            trajs = [trajs]

        # 여러 trajectory를 하나로 합침
        traj = self._merge_trajectories(trajs)

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

        returns, advantages = compute_gae(
            rewards, values, dones, self.gamma, self.gae_lambda, self.device
        )

        # 전체 데이터 크기
        total_steps = obs_t.size(0)
        
        # 평균 loss 값들을 저장
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_entropies = []
        epoch_total_losses = []

        for epoch in range(self.K_epochs):
            # 인덱스 셔플
            indices = torch.randperm(total_steps)
            
            # 미니배치로 나눠서 학습
            for start_idx in range(0, total_steps, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_steps)
                batch_indices = indices[start_idx:end_idx]
                
                # 미니배치 데이터 추출
                batch_obs = obs_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                logits, value_preds = self.policy(batch_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO loss 계산
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = (
                    torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(value_preds, batch_returns)

                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # loss 기록
                epoch_actor_losses.append(actor_loss.item())
                epoch_critic_losses.append(critic_loss.item())
                epoch_entropies.append(entropy.item())
                epoch_total_losses.append(loss.item())

        # 평균 loss 계산
        return {
            "actor_loss": sum(epoch_actor_losses) / len(epoch_actor_losses),
            "critic_loss": sum(epoch_critic_losses) / len(epoch_critic_losses),
            "entropy": sum(epoch_entropies) / len(epoch_entropies),
            "total_loss": sum(epoch_total_losses) / len(epoch_total_losses),
        }

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
        checkpoint = torch.load(
            checkpoint_file, map_location=self.device, weights_only=False
        )

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
    # 학습 로그 저장
    # -----------------------------
    def save_training_log(
        self,
        reward_history: list,
        actor_loss_history: list,
        critic_loss_history: list,
        entropy_history: list,
        start_episode: int,
        end_episode: int,
    ):
        """
        학습 로그를 JSON 파일로 저장

        Args:
            reward_history: 에피소드별 보상 기록
            actor_loss_history: Actor loss 기록
            critic_loss_history: Critic loss 기록
            entropy_history: Entropy 기록
            start_episode: 현재 세션의 시작 에피소드
            end_episode: 현재까지 진행된 마지막 에피소드
        """
        if self.checkpoint_session_dir is None:
            log.warning("체크포인트 세션 디렉토리가 설정되지 않았습니다.")
            return

        log_file = self.checkpoint_session_dir / "training_log.json"

        # 기존 로그 파일이 있으면 로드
        if log_file.exists():
            with open(log_file, "r") as f:
                existing_log = json.load(f)

            # 기존 데이터에 새로운 데이터 추가
            existing_log["episodes"].extend(list(range(start_episode, end_episode + 1)))
            existing_log["returns"].extend(reward_history)
            existing_log["actor_losses"].extend(actor_loss_history)
            existing_log["critic_losses"].extend(critic_loss_history)
            existing_log["entropies"].extend(entropy_history)

            log_data = existing_log
        else:
            # 새로운 로그 생성
            log_data = {
                "episodes": list(range(start_episode, end_episode + 1)),
                "returns": reward_history,
                "actor_losses": actor_loss_history,
                "critic_losses": critic_loss_history,
                "entropies": entropy_history,
            }

        # 로그 파일 저장
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

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
            num_episodes: 최종 목표 에피소드 번호 (체크포인트에서 이어서 하는 경우, 이 번호까지 학습)
            checkpoint_dir: 체크포인트를 저장할 디렉토리 (None이면 저장 안 함)
            checkpoint_interval: 체크포인트 저장 주기 (에피소드 단위)
            log_interval: 로그 출력 주기 (에피소드 단위)
            trajectory_save_interval: Trajectory 정보 저장 주기 (에피소드 단위)
        """
        # num_episodes 유효성 검증
        if num_episodes <= self.start_episode:
            raise ValueError(
                f"num_episodes({num_episodes})는 start_episode({self.start_episode})보다 커야 합니다. "
                f"현재 {self.start_episode}까지 진행되었으므로, num_episodes를 {self.start_episode + 1} 이상으로 설정하세요."
            )

        # 체크포인트 세션 디렉토리 초기화 (학습 시작 시 한 번만)
        if checkpoint_dir and self.checkpoint_session_dir is None:
            self.checkpoint_session_dir = Path(checkpoint_dir) / f"{today_datetime()}"
            self.checkpoint_session_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"체크포인트 세션 디렉토리 생성: {self.checkpoint_session_dir}")

        # 학습 진행 정보 로그
        episodes_to_train = num_episodes - self.start_episode
        log.info(
            f"에피소드 {self.start_episode + 1} ~ {num_episodes} "
            f"(총 {episodes_to_train}개 에피소드 학습 예정)"
        )

        reward_history = []
        # Visualization 기록용
        actor_loss_history = []
        critic_loss_history = []
        entropy_history = []

        # 전체 학습 시작 시간 기록
        total_start_time = time.time()

        # trajectory 버퍼 (buffer_size만큼 step을 쌓아두고 학습)
        traj_buffer = []
        buffer_step_count = 0  # 현재 버퍼에 쌓인 step 수

        for ep in range(self.start_episode + 1, num_episodes + 1):
            ep_start_time = time.time()

            # trajectory 수집
            traj = self._collect_trajectory()
            ep_return = sum(traj["rewards"])
            reward_history.append(ep_return)

            # 버퍼에 추가
            traj_buffer.append(traj)
            buffer_step_count += len(traj["rewards"])

            # 최고 점수 갱신 및 저장
            is_best = False
            if ep_return > self.best_score:
                self.best_score = ep_return
                is_best = True
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir, ep, is_best=True)

            # buffer_size만큼 step이 쌓였거나 마지막 에피소드인 경우 학습 수행
            should_update = (buffer_step_count >= self.buffer_size) or (ep == num_episodes)

            if should_update:
                loss_info = self.ppo_update(traj_buffer)
                # Visualization 기록 저장
                actor_loss_history.append(loss_info["actor_loss"])
                critic_loss_history.append(loss_info["critic_loss"])
                entropy_history.append(loss_info["entropy"])

                # 버퍼 초기화
                traj_buffer = []
                buffer_step_count = 0
            else:
                # 학습하지 않은 에피소드는 이전 loss 값으로 기록
                if len(actor_loss_history) > 0:
                    actor_loss_history.append(actor_loss_history[-1])
                    critic_loss_history.append(critic_loss_history[-1])
                    entropy_history.append(entropy_history[-1])
                else:
                    actor_loss_history.append(0.0)
                    critic_loss_history.append(0.0)
                    entropy_history.append(0.0)
                loss_info = {
                    "total_loss": 0.0 if len(actor_loss_history) == 1 else actor_loss_history[-1] + 0.5 * critic_loss_history[-1] - self.entropy_coef * entropy_history[-1],
                    "actor_loss": actor_loss_history[-1],
                    "critic_loss": critic_loss_history[-1],
                    "entropy": entropy_history[-1],
                }

            if ep % log_interval == 0:
                ep_elapsed_time = time.time() - ep_start_time
                update_status = f"[학습 수행]" if should_update else f"[데이터 수집 {buffer_step_count}/{self.buffer_size} steps]"
                log.info(
                    f"[Episode {ep}] {update_status} return = {ep_return:.3f}, len = {len(traj['rewards'])}, "
                    f"time = {ep_elapsed_time:.2f}s, "
                    f"loss = {loss_info['total_loss']:.4f} "
                    f"(actor: {loss_info['actor_loss']:.4f}, critic: {loss_info['critic_loss']:.4f}, entropy: {loss_info['entropy']:.4f})"
                )

            # trajectory 정보 저장
            if checkpoint_dir and ep % trajectory_save_interval == 0:
                self.save_trajectory_info(checkpoint_dir, ep, traj)

            # 체크포인트 저장 (주기적)
            if checkpoint_dir and ep % checkpoint_interval == 0:
                # 이미 best로 저장했으면 중복 저장 방지 (선택사항이지만 파일IO 줄이기 위해)
                if not is_best:
                    self.save_checkpoint(checkpoint_dir, ep)
                
                # 학습 로그도 주기적으로 저장
                self.save_training_log(
                    reward_history=reward_history,
                    actor_loss_history=actor_loss_history,
                    critic_loss_history=critic_loss_history,
                    entropy_history=entropy_history,
                    start_episode=self.start_episode + 1,
                    end_episode=ep,
                )

        # 학습 종료 시 최종 체크포인트 저장
        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir, num_episodes)

            # 학습 로그 저장 (이어서 학습하는 경우 기존 로그에 추가)
            log_file = self.checkpoint_session_dir / "training_log.json"

            # 기존 로그 파일이 있으면 로드
            if log_file.exists():
                with open(log_file, "r") as f:
                    existing_log = json.load(f)

                # 기존 데이터에 새로운 데이터 추가
                existing_log["episodes"].extend(
                    list(range(self.start_episode + 1, num_episodes + 1))
                )
                existing_log["returns"].extend(reward_history)
                existing_log["actor_losses"].extend(actor_loss_history)
                existing_log["critic_losses"].extend(critic_loss_history)
                existing_log["entropies"].extend(entropy_history)

                log_data = existing_log
                log.info(f"기존 학습 로그에 추가: {log_file}")
            else:
                # 새로운 로그 생성
                log_data = {
                    "episodes": list(range(self.start_episode + 1, num_episodes + 1)),
                    "returns": reward_history,
                    "actor_losses": actor_loss_history,
                    "critic_losses": critic_loss_history,
                    "entropies": entropy_history,
                }
                log.info(f"새로운 학습 로그 생성: {log_file}")

            # 로그 파일 저장
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=2)

            log.info(f"학습 로그 저장 완료: {log_file}")

        # 전체 학습 시간 계산 및 출력
        total_elapsed_time = time.time() - total_start_time
        hours = int(total_elapsed_time // 3600)
        minutes = int((total_elapsed_time % 3600) // 60)
        seconds = total_elapsed_time % 60

        log.info(
            f"학습 완료: 에피소드 {self.start_episode + 1} ~ {num_episodes} "
            f"(총 {num_episodes - self.start_episode}개 에피소드 학습 완료)"
        )
        log.info(
            f"총 학습 시간: {hours}시간 {minutes}분 {seconds:.2f}초 "
            f"(총 {total_elapsed_time:.2f}초)"
        )

        return reward_history

    # -----------------------------
    # Greedy 평가
    # -----------------------------
    def evaluate_greedy(
        self, doc_index: int = None, use_cache=False, save_to_cache=False
    ):
        """
        학습된 정책을 greedy로 평가:
        - 각 step에서 argmax(logits)로 행동 선택

        Args:
            doc_index: 평가할 문서의 인덱스 (None이면 환경의 기본 reset() 사용)

        Returns:
            dict: 평가 결과 정보 (최종 점수, 선택된 액션 등)
        """

        self.env.use_cache = use_cache
        self.env.save_to_cache = save_to_cache

        # 문서 인덱스가 지정된 경우
        if doc_index is not None:
            # 문서 인덱스 유효성 검사
            if doc_index < 0 or doc_index >= len(self.env.documents):
                raise ValueError(
                    f"유효하지 않은 문서 인덱스: {doc_index}. "
                    f"유효 범위: 0 ~ {len(self.env.documents) - 1}"
                )

            # 환경 초기화
            self.env.current_step = 0
            self.env.used_actions = set()
            self.env.action_history = []

            # 특정 문서 선택
            self.env.doc_index = doc_index
            base_doc = self.env.documents[doc_index]
            self.env.current_text = base_doc.text

            # 초기 점수 계산
            self.env.current_score = self.env.judge.score(self.env.current_text)
            state = self.env._scores_to_state(self.env.current_score)
            text = self.env.current_text

            log.info(f"[Eval] 문서 인덱스: {doc_index}")
        else:
            # 기본 reset() 사용
            state, text = self.env.reset()

        # 초기 점수를 자동으로 포맷팅 (DocumentScore의 필드를 동적으로 가져옴)
        score_fields = [f.name for f in fields(self.env.current_score)]
        score_lines = [
            f"  - {field:22s}: {getattr(self.env.current_score, field):.2f}"
            for field in score_fields
        ]
        score_text = "\n".join(score_lines)

        log.info(
            f"""[Eval] 초기 문서 내용:
{"-" * 60}
{text}
{"-" * 60}
초기 점수:
{score_text}"""
        )

        actions_taken = []
        rewards_received = []
        last_action_idx = -1
        initial_score = self.env.current_score

        for t in range(self.max_steps):
            obs = self._build_obs(
                state=state,
                step_idx=t,
                last_action_idx=last_action_idx,
            )
            obs_t = obs.unsqueeze(0)

            with torch.no_grad():
                logits, value = self.policy(obs_t)
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                action = torch.argmax(logits, dim=-1)

            action_idx = int(action.item())
            actions_taken.append(action_idx)
            action_name = self.env.actions[action_idx]

            # 액션 확률 분포 출력
            action_prob_log = (
                f"\n[Step {t+1}] 액션 확률 분포 (Value: {value.item():.3f}):\n"
            )
            for act_name, prob in zip(self.env.actions, probs):
                bar = "*" * int(prob * 25)
                marker = " <-- 선택됨" if act_name == action_name else ""
                action_prob_log += f"  {act_name:20s}: {prob:.3f} {bar}{marker}\n"
            log.info(action_prob_log)

            before_text = self.env.current_text
            next_state, reward, done, info = self.env.step(action_idx)
            rewards_received.append(reward)

            log.info(f"[Step {t+1}] action = {action_name}, reward = {reward:.3f}")

            # prev_scores 포맷팅
            prev_scores = info.get("prev_scores")
            if prev_scores:
                score_fields = [f.name for f in fields(prev_scores)]
                score_lines = [
                    f"  - {field:22s}: {getattr(prev_scores, field):.2f}"
                    for field in score_fields
                ]
                score_text = "\n".join(score_lines)
                log.info(f"Prev scores:\n{score_text}")

            # new_scores 포맷팅
            new_scores = info.get("new_scores")
            if new_scores:
                score_fields = [f.name for f in fields(new_scores)]
                score_lines = [
                    f"  - {field:22s}: {getattr(new_scores, field):.2f}"
                    for field in score_fields
                ]
                score_text = "\n".join(score_lines)
                log.info(f"New  scores:\n{score_text}")

            log.info(f"[Before]\n{before_text}")
            log.info(f"[After]\n{self.env.current_text}")

            last_action_idx = action_idx
            state = next_state
            if done:
                log.info(
                    f"[Eval] 종료 (reason={info.get('reason', 'unknown')}, step={t+1})"
                )
                break

        # 최종 점수 포맷팅
        final_score = self.env.current_score
        score_fields = [f.name for f in fields(final_score)]
        score_lines = [
            f"  - {field:22s}: {getattr(final_score, field):.2f}"
            for field in score_fields
        ]
        score_text = "\n".join(score_lines)

        log.info(f"[Eval] 최종 점수:\n{score_text}")
        log.info(f"선택된 액션 인덱스 시퀀스: {actions_taken}")
        log.info(f"선택된 액션 시퀀스: {[self.env.actions[a] for a in actions_taken]}")
        log.info(f"총 보상: {sum(rewards_received):.3f}")

        # 평가 결과 반환
        result = {
            "doc_index": (
                doc_index
                if doc_index is not None
                else getattr(self.env, "doc_index", None)
            ),
            "initial_score": initial_score,
            "final_score": self.env.current_score,
            "actions_taken": actions_taken,
            "action_names": [self.env.actions[a] for a in actions_taken],
            "rewards": rewards_received,
            "total_reward": sum(rewards_received),
            "num_steps": len(actions_taken),
            "final_text": self.env.current_text,
        }

        return result

    # -----------------------------
    # 정책 시각화
    # -----------------------------
    def show_policy(self):
        """학습된 정책의 액션 확률 분포 확인"""
        state, _ = self.env.reset()

        log.info(
            f"""현재 문서 점수:
structure:           {state[0]:.2f}
length:              {state[1]:.2f}
academic_style:      {state[2]:.2f}
information_density: {state[3]:.2f}
clarity:             {state[4]:.2f}
overall:             {state[5]:.2f}\n"""
        )

        obs = self._build_obs(state, 0, -1)
        with torch.no_grad():
            logits, value = self.policy(obs.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Step 1의 액션 확률 출력
        action_log = f"Step 1 액션 확률 (Value: {value.item():.3f}):\n"
        for action, prob in zip(self.env.actions, probs):
            bar = "*" * int(prob * 25)
            action_log += f"  {action:20s}: {prob:.3f} {bar}\n"

        log.info(action_log)
