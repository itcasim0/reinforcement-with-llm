"""
edit_document_offline.py에는 사전에 미리 LLM을 통해 도출된 결과를 가지고 학습과 평가하는 코드가 공존하므로 참고
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
from src.environments.editing_env.offline_env import OfflineEditingEnv
from src.environments.editing_env.components.component import DocumentScore
from src.dataloader.offline_loader import OfflineDocumentLoader
from src.methods.ppo.runner import PPORunner
from src.config.paths import LOGS_DIR, DATA_DIR
from src.utils.logger_factory import log

# 재현을 위한 seed 설정
SEED = 42
# 재현을 위한 랜덤 시드 고정
random.seed(SEED)
torch.manual_seed(SEED)

# 오프라인 데이터 경로 (디렉토리)
OFFLINE_DATA_PATH = DATA_DIR / "paper_data" / "offline"

# ========== parameters for environment ==========
TERMINAL_THRESHOLD = 9.5  # 문서의 종합 품질 점수에 따라 종료할 한계점
REPEAT_PENALTY = 0.5  # 반복 액션에 대한 패널티 정도
EDITOR_MODEL = "qwen/qwen3-8b"  # 오프라인 환경에서는 실제로 LLM을 사용하지 않음

# 학습 시 LLM 비용에 대한 가중치로, COST_LAMBDA만큼 step마다 사용한 실제 비용에 곱하여 패널티 부과
# NOTE: 현재 LLM 비용 패널티는 고정해두었으니 튜닝하지 말 것
COST_LAMBDA = 1.0

STEP_PENALTY = 0.09  # step 하나 당 패널티 (ex) reward -= 2step * 패널티)

MAX_STEPS = 3  # 한 1 episode당 허용할 최대 step 수

USE_SINGLE_SEQUENCE = True  # 오버피팅 모드 (첫 번째 시퀀스만 사용)

# ========== parameters for train ==========
CHECKPOINT_DIR = None  # 학습 재개를 위한 설정 (저장된 체크포인트 디렉토리 경로)
SAVE_CHECKPOINT_DIR = LOGS_DIR / "checkpoints" / "ppo_offline"
CHECKPOINT_INTERVAL = 100
LOG_INTERVAL = 100

BUFFER_SIZE = 3  # 학습 전에 모을 step 수
BATCH_SIZE = 3  # 미니배치 크기
K_EPOCHS = 3  # BUFFER_SIZE만큼 쌓인 후 update하는 횟수

NUM_EPISODES = 1000

# estimator에서 사용하는 값
GAMMA = 0.95
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
CLIP_EPS = 0.2
LR = 3e-4


def main():

    # load data
    log.info("데이터 로드")
    dataloader = OfflineDocumentLoader(jsonl_path=OFFLINE_DATA_PATH)

    # 강화학습 환경 구성
    log.info("강화학습 환경 구성")
    env = OfflineEditingEnv(
        dataloader=dataloader,
        max_steps=MAX_STEPS,
        terminal_threshold=TERMINAL_THRESHOLD,
        cost_lambda=COST_LAMBDA,
        repeat_penalty=REPEAT_PENALTY,
        editor_model=EDITOR_MODEL,
        use_single_sequence=USE_SINGLE_SEQUENCE,
        fixed_sequence_idx=0,
    )

    # 강화학습 정책 구성
    log.info("강화학습 정책 구성")
    len_scores = len(fields(DocumentScore))  # 평가 지표 (state)의 개수
    runner = PPORunner(
        env=env,
        max_steps=MAX_STEPS,
        state_dim=len_scores
        + 1
        + env.num_actions,  # scores + step + last_action_one_hot
        num_actions=env.num_actions,
        lr=LR,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        entropy_coef=ENTROPY_COEF,
        clip_eps=CLIP_EPS,
        K_epochs=K_EPOCHS,  # PPO 업데이트 반복 횟수
        buffer_size=BUFFER_SIZE,  # 학습 전에 모을 step 수
        batch_size=BATCH_SIZE,  # 미니배치 크기
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
    )

    # 평가 시작
    log.info("평가")
    runner.evaluate_greedy()
    return


if __name__ == "__main__":
    main()
