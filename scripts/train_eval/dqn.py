"""
edit_document_dqn.py - DQN을 사용한 문서 편집 학습 및 평가 스크립트

DQN (Deep Q-Network)의 핵심 특징:
- Experience Replay: 과거 경험을 재사용하여 학습 효율성 향상
- Target Network: 고정된 타겟으로 학습 안정화
- Epsilon-greedy: 탐색과 활용의 균형
- Off-policy 학습: 수집한 경험을 여러 번 재사용 가능
"""

import sys
from pathlib import Path
import random
from dataclasses import fields

import torch

# 프로젝트 루트를 Python 경로에 추가
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(1, str(root_dir / "src"))

# internal
from src.dataloader.reconstruct_loader import DomesticReconstructDataLoader
from src.environments.editing_env.base_env import EditingEnv
from src.environments.editing_env.components.component import DocumentScore
from src.methods.dqn.runner import DQNRunner

from src.config.paths import LOGS_DIR, DATA_DIR
from src.utils.logger_factory import log

# 재현을 위한 seed 설정
SEED = 42
# 재현을 위한 랜덤 시드 고정
random.seed(SEED)
torch.manual_seed(SEED)

# 입력 데이터셋 json 파일
INPUT_DATA = DATA_DIR / "paper_data" / "reconstruct"

# ========== parameters for environment ==========
TERMINAL_THRESHOLD = 9.5  # 문서의 종합 품질 점수에 따라 종료할 한계점
REAPEAT_PANELTY = 0.5  # 반복 액션에 대한 패널티 정도
# EDITOR_MODEL = "google/gemma-3-27b-it"  # 액션에 대한 LLM(or SLM)
# EDITOR_MODEL = "qwen/qwen3-8b"  # 조금 더 성능이 좋지 않은 모델로 실험하기 위함
EDITOR_MODEL = "google/gemma-3n-e4b-it"  # qwen3-8b는 thinking모델로 스스로 생각하는 시간 때문에 너무 느림

# 학습 시 LLM 비용에 대한 가중치로, COST_LAMBDA만큼 step마다 사용한 실제 비용에 곱하여 패널티 부과
# NOTE: 현재 LLM 비용 패널티는 고정해두었으니 튜닝하지 말 것
COST_LAMBDA = 1.0

STEP_PENLTY = 0.09  # step 하나 당 패널티 (ex) reward -= 2step * 패널티)

MAX_STEPS = 5  # 한 1 episode당 허용할 최대 step 수

# ========== parameters for train ==========
# CHECKPOINT_DIR = r"D:\SMC\projects\reinforcement-with-llm\logs\checkpoints\20251204T133523"  # 학습 재개를 위한 설정 (저장된 체크포인트 디렉토리 경로)
CHECKPOINT_DIR = None
SAVE_CHECKPOINT_DIR = LOGS_DIR / "checkpoints" / "dqn"
CHECKPOINT_INTERVAL = 10
LOG_INTERVAL = 10
TRAJECTORY_SAVE_INTERVAL = 50  # DQN은 trajectory 저장을 덜 자주 함

NUM_EPISODES = 1000

# DQN 특유의 하이퍼파라미터
GAMMA = 0.95  # 할인율
EPSILON_START = 1.0  # 초기 탐색 확률 (100% 탐색으로 시작)
EPSILON_END = 0.01  # 최종 탐색 확률 (1% 탐색 유지)
EPSILON_DECAY = 0.995  # Epsilon 감소율 (매 에피소드마다 곱함)

BUFFER_SIZE = 10000  # Experience Replay Buffer 크기
BATCH_SIZE = 32  # 학습 배치 크기
TARGET_UPDATE_FREQ = 10  # Target network 업데이트 주기 (에피소드 단위)

LEARNING_RATE = 1e-3  # DQN은 일반적으로 PPO/A2C보다 높은 학습률 사용


def main():

    # load data
    log.info("데이터 로드")
    dataloader = DomesticReconstructDataLoader(json_path=INPUT_DATA)

    # 강화학습 환경 구성
    log.info("강화학습 환경 구성")
    env = EditingEnv(
        dataloader=dataloader,
        max_steps=MAX_STEPS,
        terminal_threshold=TERMINAL_THRESHOLD,
        cost_lambda=COST_LAMBDA,
        repeat_penalty=REAPEAT_PANELTY,  # 반복 액션에 대한 패널티 정도
        editor_model=EDITOR_MODEL,
    )

    # 강화학습 정책 구성 (DQN)
    log.info("강화학습 정책 구성 (DQN)")
    len_scores = len(fields(DocumentScore))  # 평가 지표 (state)의 개수
    runner = DQNRunner(
        env=env,
        max_steps=MAX_STEPS,
        state_dim=len_scores
        + 1
        + env.num_actions,  # scores + step + last_action_one_hot
        num_actions=env.num_actions,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
    )

    # 체크포인트에서 재개
    if CHECKPOINT_DIR:
        try:
            runner.load_checkpoint(CHECKPOINT_DIR)
            log.info(f"체크포인트에서 학습 재개: {CHECKPOINT_DIR}")
        except FileNotFoundError as e:
            log.error(f"체크포인트 로드 실패: {e}")
            return

    # 학습 시작
    log.info("학습 시작")
    runner.train(
        num_episodes=NUM_EPISODES,
        checkpoint_dir=SAVE_CHECKPOINT_DIR,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        log_interval=LOG_INTERVAL,
        trajectory_save_interval=TRAJECTORY_SAVE_INTERVAL,
    )

    # 평가 시작
    log.info("평가")
    runner.evaluate_greedy()
    return


if __name__ == "__main__":
    main()
