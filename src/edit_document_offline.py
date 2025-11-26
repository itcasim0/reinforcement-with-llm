"""
edit_document_offline.py에는 사전에 미리 LLM을 통해 도출된 결과를 가지고 학습과 평가하는 코드가 공존하므로 참고
"""

import sys
from pathlib import Path

# src 디렉토리를 sys.path에 추가
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import random

import torch

# internal
from dataloader.reconstruct_loader import ReconstructDataLoader
from environments.editing_env.env import OfflineEditingEnv
from methods.ppo import PPORunner

from config.paths import LOGS_DIR
from utils.logger_factory import log

# 재현을 위한 seed 설정
SEED = 42

# parameters for environment
TERMINAL_THRESHOLD = 9.5  # 문서의 종합 품질 점수에 따라 종료할 한계점
REAPEAT_PANELTY = 0.3  # 반복 액션에 대한 패널티 정도
# EDITOR_MODEL = "google/gemma-3-27b-it"  # 액션에 대한 LLM(or SLM)
EDITOR_MODEL = "qwen/qwen3-8b"  # 조금 더 성능이 좋지 않은 모델로 실험하기 위함

# parameters for train
CHECKPOINT_DIR = None  # 학습 재개를 위한 설정 (저장된 체크포인트 디렉토리 경로)
SAVE_CHECKPOINT_DIR = LOGS_DIR / "checkpoints"
CHECKPOINT_INTERVAL = 100
LOG_INTERVAL = 100
NUM_EPISODES = 1000

# 재현을 위한 랜덤 시드 고정
random.seed(SEED)
torch.manual_seed(SEED)


def main():

    # load data
    log.info("데이터 로드")
    dataloader = ReconstructDataLoader()
    documents = dataloader.get_reconstructed_text(max_docs=5)

    # 강화학습 환경 구성
    # TODO: 현재는 데이터가 하나로 고정되어있어서 documents를 사실상 안받아도 됨
    log.info("강화학습 환경 구성")
    env = OfflineEditingEnv(
        documents=documents,
        max_steps=3,
        terminal_threshold=TERMINAL_THRESHOLD,
        cost_lambda=1.0,
        repeat_penalty=REAPEAT_PANELTY,  # 반복 액션에 대한 패널티 정도
        editor_model=EDITOR_MODEL,
    )

    # 강화학습 정책 구성
    log.info("강화학습 정책 구성")
    runner = PPORunner(
        env=env,
        max_steps=3,
        state_dim=4 + 1 + env.num_actions,  # g,r,c,o + step + last_action_one_hot
        num_actions=env.num_actions,
        gamma=0.95,
        lr=3e-4,
        clip_eps=0.2,
        K_epochs=3,  # 한번의 에피소드 내에서 수행할 신경망 모델 학습 epochs
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
    runner.evaluate_greedy(max_steps=3)

    return


if __name__ == "__main__":
    main()
