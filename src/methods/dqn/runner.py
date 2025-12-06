from typing import Tuple
from pathlib import Path
import json
from dataclasses import fields
import time
import random

# external
import torch
import torch.nn as nn
import torch.optim as optim

# internal
from environments.editing_env.base_env import EditingEnv
from environments.editing_env.offline_env import OfflineEditingEnv

from utils.util import today_datetime
from utils.logger_factory import log

from .policy import DQNPolicy
from .replay_buffer import ReplayBuffer


class DQNRunner:
    """
    DQN (Deep Q-Network) 학습 러너
    
    - 환경에서 경험 수집 및 Replay Buffer에 저장
    - Q-learning으로 Q-value 업데이트
    - Target network로 학습 안정화
    
    DQN의 핵심 특징:
    1. Experience Replay: 과거 경험을 재사용
    2. Target Network: 고정된 타겟으로 학습 안정화
    3. Epsilon-greedy: 탐색과 활용의 균형

    Args:
        env (EditingEnv): 강화학습 환경 (문서 편집 환경)
        max_steps (int): 에피소드당 최대 스텝 수
        num_actions (int): 가능한 액션의 개수
        gamma (float): 할인율 (discount factor). 기본값 0.99
        epsilon_start (float): 초기 epsilon (탐색 확률). 기본값 1.0
        epsilon_end (float): 최종 epsilon. 기본값 0.01
        epsilon_decay (float): epsilon 감소율. 기본값 0.995
        lr (float): 학습률. 기본값 1e-3
        buffer_size (int): Replay buffer 크기. 기본값 10000
        batch_size (int): 학습 배치 크기. 기본값 32
        target_update_freq (int): Target network 업데이트 주기. 기본값 10
    """

    def __init__(
        self,
        env: EditingEnv | OfflineEditingEnv,
        max_steps: int,
        state_dim: int,
        num_actions: int,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        lr: float = 1e-3,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 10,
    ):
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"디바이스 설정: {self.device}")

        # 환경 초기화
        self.env: EditingEnv | OfflineEditingEnv = env
        self.max_steps = max_steps

        # DQN 설계
        self.num_actions = num_actions
        self.state_dim = state_dim

        # 하이퍼파라미터
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Q-network와 Target network
        self.q_network = DQNPolicy(state_dim=state_dim, num_actions=num_actions).to(
            self.device
        )
        self.target_network = DQNPolicy(
            state_dim=state_dim, num_actions=num_actions
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network는 evaluation mode

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # 체크포인트 관련
        self.start_episode = 0
        self.checkpoint_session_dir = None
        self.best_score = -float("inf")
        self.last_logged_episode = 0  # 마지막으로 로그 저장한 에피소드

    def _build_obs(
        self,
        state: Tuple[float, float, float, float, float, float],
        step_idx: int,
        last_action_idx: int,
    ) -> torch.Tensor:
        """
        관측 벡터 생성
        
        state: (structure, length, academic_style, information_density, clarity, overall)
        step_idx: 현재 step 인덱스
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

        step_norm = torch.tensor(
            [step_idx / self.max_steps],
            dtype=torch.float32,
            device=self.device,
        )

        last_onehot = torch.zeros(
            self.num_actions, dtype=torch.float32, device=self.device
        )
        if 0 <= last_action_idx < self.num_actions:
            last_onehot[last_action_idx] = 1.0

        obs = torch.cat([base, step_norm, last_onehot], dim=0)
        return obs

    def select_action(self, obs: torch.Tensor, epsilon: float) -> int:
        """
        Epsilon-greedy 정책으로 액션 선택
        
        Args:
            obs: 관측 벡터
            epsilon: 탐색 확률
            
        Returns:
            선택된 액션 인덱스
        """
        if random.random() < epsilon:
            # 탐색: 무작위 액션
            return random.randrange(self.num_actions)
        else:
            # 활용: Q-value가 가장 높은 액션
            with torch.no_grad():
                q_values = self.q_network(obs.unsqueeze(0))
                return int(q_values.argmax(dim=1).item())

    def _collect_trajectory(self):
        """
        에피소드 하나를 rollout하여 경험을 Replay Buffer에 저장
        """
        states = []
        actions = []
        rewards = []
        infos = []

        state, _ = self.env.reset()
        last_action_idx = -1
        ep_return = 0.0

        for t in range(self.max_steps):
            obs = self._build_obs(
                state=state, step_idx=t, last_action_idx=last_action_idx
            )

            # Epsilon-greedy로 액션 선택
            action_idx = self.select_action(obs, self.epsilon)

            next_state, reward, done, info = self.env.step(action_idx)
            ep_return += reward

            # 다음 관측 벡터 생성
            next_obs = self._build_obs(
                state=next_state, step_idx=t + 1, last_action_idx=action_idx
            )

            # Replay Buffer에 저장
            self.replay_buffer.push(obs, action_idx, reward, next_obs, done)

            # 기록 (trajectory 정보 저장용)
            states.append(state)
            actions.append(action_idx)
            rewards.append(reward)
            infos.append(info)

            last_action_idx = action_idx
            state = next_state

            if done:
                break

        traj = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "infos": infos,
        }

        return traj, ep_return

    def dqn_update(self):
        """
        DQN 업데이트 수행
        
        Loss: MSE(Q(s,a), r + γ * max Q_target(s',a'))
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}

        # Replay Buffer에서 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # 현재 Q-values: Q(s,a)
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 타겟 Q-values: r + γ * max Q_target(s',a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Loss 계산 및 업데이트
        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_target_network(self):
        """Target network를 Q-network로 업데이트"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_checkpoint(self, checkpoint_dir: str, episode: int, is_best: bool = False):
        """체크포인트 저장"""
        if self.checkpoint_session_dir is None:
            log.warning("체크포인트 세션 디렉토리가 설정되지 않았습니다.")
            return

        checkpoint_file = self.checkpoint_session_dir / f"checkpoint_ep{episode}.pt"

        checkpoint = {
            "episode": episode,
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "gamma": self.gamma,
            "best_score": self.best_score,
        }

        torch.save(checkpoint, checkpoint_file)

        latest_file = self.checkpoint_session_dir / "latest_checkpoint.txt"
        with open(latest_file, "w") as f:
            f.write(str(checkpoint_file))

        if is_best:
            best_file = self.checkpoint_session_dir / "best_checkpoint.txt"
            with open(best_file, "w") as f:
                f.write(str(checkpoint_file))

    def load_checkpoint(self, checkpoint_path: str, load_best: bool = False):
        """체크포인트 로드"""
        path = Path(checkpoint_path)

        if path.is_dir():
            if load_best:
                ptr_file = path / "best_checkpoint.txt"
            else:
                ptr_file = path / "latest_checkpoint.txt"

            if ptr_file.exists():
                with open(ptr_file, "r") as f:
                    checkpoint_file = Path(f.read().strip())
            else:
                if load_best:
                    raise FileNotFoundError(f"Best 체크포인트를 찾을 수 없습니다: {ptr_file}")

                checkpoints = sorted(path.glob("checkpoint_ep*.pt"))
                if not checkpoints:
                    raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {path}")
                checkpoint_file = checkpoints[-1]
        else:
            checkpoint_file = path

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"체크포인트가 존재하지 않습니다: {checkpoint_file}")

        log.info(f"체크포인트 로드: {checkpoint_file}")
        checkpoint = torch.load(
            checkpoint_file, map_location=self.device, weights_only=False
        )

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.start_episode = checkpoint["episode"]

        if "best_score" in checkpoint:
            self.best_score = checkpoint["best_score"]

        self.checkpoint_session_dir = checkpoint_file.parent
        log.info(f"체크포인트 세션 디렉토리: {self.checkpoint_session_dir}")
        log.info(
            f"에피소드 {self.start_episode}부터 재개 (epsilon={self.epsilon:.3f}, best_score={self.best_score})"
        )

        return self.start_episode

    def save_training_log(
        self,
        reward_history: list,
        loss_history: list,
        epsilon_history: list,
        start_episode: int,
        end_episode: int,
    ):
        """학습 로그 저장"""
        if self.checkpoint_session_dir is None:
            log.warning("체크포인트 세션 디렉토리가 설정되지 않았습니다.")
            return

        log_file = self.checkpoint_session_dir / "training_log.json"

        if log_file.exists():
            with open(log_file, "r") as f:
                existing_log = json.load(f)

            existing_log["episodes"].extend(list(range(start_episode, end_episode + 1)))
            existing_log["returns"].extend(reward_history)
            existing_log["losses"].extend(loss_history)
            existing_log["epsilons"].extend(epsilon_history)

            log_data = existing_log
        else:
            log_data = {
                "episodes": list(range(start_episode, end_episode + 1)),
                "returns": reward_history,
                "losses": loss_history,
                "epsilons": epsilon_history,
            }

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    def save_trajectory_info(self, checkpoint_dir: str, episode: int, traj: dict):
        """Trajectory 정보 저장"""
        if self.checkpoint_session_dir is None:
            log.warning("체크포인트 세션 디렉토리가 설정되지 않았습니다.")
            return

        traj_dir = self.checkpoint_session_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)

        traj_file = traj_dir / f"episode_{episode}.json"

        traj_data = {
            "episode": episode,
            "total_return": sum(traj["rewards"]),
            "num_steps": len(traj["rewards"]),
            "steps": [],
        }

        for step_idx in range(len(traj["rewards"])):
            info = traj["infos"][step_idx] if step_idx < len(traj["infos"]) else {}
            serializable_info = {}

            for key, value in info.items():
                if hasattr(value, "__dict__"):
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
                "info": serializable_info,
            }
            traj_data["steps"].append(step_info)

        with open(traj_file, "w", encoding="utf-8") as f:
            json.dump(traj_data, f, ensure_ascii=False, indent=2)

    def train(
        self,
        num_episodes: int = 1000,
        checkpoint_dir: str = None,
        checkpoint_interval: int = 50,
        log_interval: int = 10,
        trajectory_save_interval: int = 50,
    ):
        """
        DQN 학습 루프
        
        Args:
            num_episodes: 최종 목표 에피소드 번호
            checkpoint_dir: 체크포인트 저장 디렉토리
            checkpoint_interval: 체크포인트 저장 주기
            log_interval: 로그 출력 주기
            trajectory_save_interval: Trajectory 저장 주기
        """
        if num_episodes <= self.start_episode:
            raise ValueError(
                f"num_episodes({num_episodes})는 start_episode({self.start_episode})보다 커야 합니다."
            )

        if checkpoint_dir and self.checkpoint_session_dir is None:
            self.checkpoint_session_dir = Path(checkpoint_dir) / f"{today_datetime()}"
            self.checkpoint_session_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"체크포인트 세션 디렉토리 생성: {self.checkpoint_session_dir}")

        episodes_to_train = num_episodes - self.start_episode
        log.info(
            f"에피소드 {self.start_episode + 1} ~ {num_episodes} "
            f"(총 {episodes_to_train}개 에피소드 학습 예정)"
        )

        reward_history = []
        loss_history = []
        epsilon_history = []

        total_start_time = time.time()

        for ep in range(self.start_episode + 1, num_episodes + 1):
            ep_start_time = time.time()

            # Trajectory 수집
            traj, ep_return = self._collect_trajectory()
            reward_history.append(ep_return)
            epsilon_history.append(self.epsilon)

            # DQN 업데이트
            loss_info = self.dqn_update()
            loss_history.append(loss_info["loss"])

            # Epsilon 감소
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Target network 업데이트
            if ep % self.target_update_freq == 0:
                self.update_target_network()

            # 최고 점수 갱신
            is_best = False
            if ep_return > self.best_score:
                self.best_score = ep_return
                is_best = True
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir, ep, is_best=True)

            if ep % log_interval == 0:
                ep_elapsed_time = time.time() - ep_start_time
                log.info(
                    f"[Episode {ep}] return = {ep_return:.3f}, len = {len(traj['rewards'])}, "
                    f"time = {ep_elapsed_time:.2f}s, "
                    f"loss = {loss_info['loss']:.4f}, epsilon = {self.epsilon:.3f}, "
                    f"buffer = {len(self.replay_buffer)}"
                )

            # Trajectory 저장
            if checkpoint_dir and ep % trajectory_save_interval == 0:
                self.save_trajectory_info(checkpoint_dir, ep, traj)

            # 체크포인트 저장
            if checkpoint_dir and ep % checkpoint_interval == 0:
                if not is_best:
                    self.save_checkpoint(checkpoint_dir, ep)

                # 마지막 로그 저장 이후의 새로운 데이터만 저장
                episodes_to_save = ep - self.last_logged_episode
                self.save_training_log(
                    reward_history=reward_history[-episodes_to_save:],
                    loss_history=loss_history[-episodes_to_save:],
                    epsilon_history=epsilon_history[-episodes_to_save:],
                    start_episode=self.last_logged_episode + 1,
                    end_episode=ep,
                )
                self.last_logged_episode = ep

        # 학습 종료
        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir, num_episodes)

            # 마지막 checkpoint_interval 이후의 데이터만 저장
            last_checkpoint_ep = (num_episodes // checkpoint_interval) * checkpoint_interval
            if last_checkpoint_ep < num_episodes:
                # 마지막 checkpoint 이후의 새로운 데이터가 있는 경우에만 저장
                episodes_since_last_checkpoint = num_episodes - last_checkpoint_ep
                self.save_training_log(
                    reward_history=reward_history[-episodes_since_last_checkpoint:],
                    loss_history=loss_history[-episodes_since_last_checkpoint:],
                    epsilon_history=epsilon_history[-episodes_since_last_checkpoint:],
                    start_episode=last_checkpoint_ep + 1,
                    end_episode=num_episodes,
                )
                log.info(f"마지막 {episodes_since_last_checkpoint}개 에피소드 로그 추가 저장")

        total_elapsed_time = time.time() - total_start_time
        hours = int(total_elapsed_time // 3600)
        minutes = int((total_elapsed_time % 3600) // 60)
        seconds = total_elapsed_time % 60

        log.info(
            f"학습 완료: 에피소드 {self.start_episode + 1} ~ {num_episodes} "
            f"(총 {num_episodes - self.start_episode}개 에피소드)"
        )
        log.info(
            f"총 학습 시간: {hours}시간 {minutes}분 {seconds:.2f}초"
        )

        return reward_history

    def evaluate_greedy(
        self, doc_index: int = None, use_cache=False, save_to_cache=False
    ):
        """
        학습된 Q-network를 greedy로 평가
        가장 최근 학습된 best checkpoint를 자동으로 로드하여 평가
        """
        # best checkpoint 자동 로드
        if self.checkpoint_session_dir is not None:
            try:
                log.info("Best checkpoint 로드 시도...")
                self.load_checkpoint(str(self.checkpoint_session_dir), load_best=True)
            except FileNotFoundError as e:
                log.warning(f"Best checkpoint를 찾을 수 없습니다: {e}")
                log.info("현재 메모리의 모델로 평가를 진행합니다.")
        
        self.env.use_cache = use_cache
        self.env.save_to_cache = save_to_cache

        if doc_index is not None:
            if doc_index < 0 or doc_index >= len(self.env.documents):
                raise ValueError(
                    f"유효하지 않은 문서 인덱스: {doc_index}. "
                    f"유효 범위: 0 ~ {len(self.env.documents) - 1}"
                )

            self.env.current_step = 0
            self.env.used_actions = set()
            self.env.action_history = []
            self.env.doc_index = doc_index
            base_doc = self.env.documents[doc_index]
            self.env.current_text = base_doc.text
            self.env.current_score = self.env.judge.score(self.env.current_text)
            state = self.env._scores_to_state(self.env.current_score)
            text = self.env.current_text

            log.info(f"[Eval] 문서 인덱스: {doc_index}")
        else:
            state, text = self.env.reset()

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

            with torch.no_grad():
                q_values = self.q_network(obs.unsqueeze(0)).squeeze()
                action_idx = int(q_values.argmax().item())

            actions_taken.append(action_idx)
            action_name = self.env.actions[action_idx]

            # Q-values 출력
            q_val_log = f"\n[Step {t+1}] Q-values:\n"
            for act_name, q_val in zip(self.env.actions, q_values.cpu().numpy()):
                bar = "*" * int((q_val + 10) / 20 * 25) if q_val > -10 else ""
                marker = " <-- 선택됨" if act_name == action_name else ""
                q_val_log += f"  {act_name:20s}: {q_val:7.3f} {bar}{marker}\n"
            log.info(q_val_log)

            before_text = self.env.current_text
            next_state, reward, done, info = self.env.step(action_idx)
            rewards_received.append(reward)

            log.info(f"[Step {t+1}] action = {action_name}, reward = {reward:.3f}")

            prev_scores = info.get("prev_scores")
            if prev_scores:
                score_fields = [f.name for f in fields(prev_scores)]
                score_lines = [
                    f"  - {field:22s}: {getattr(prev_scores, field):.2f}"
                    for field in score_fields
                ]
                score_text = "\n".join(score_lines)
                log.info(f"Prev scores:\n{score_text}")

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
                log.info(f"[Eval] 종료 (reason={info.get('reason', 'unknown')}, step={t+1})")
                break

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
