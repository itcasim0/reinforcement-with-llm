"""
edit_document_offline.py에는 사전에 미리 LLM을 통해 도출된 결과를 가지고 학습과 평가하는 코드가 공존하므로 참고
"""

import sys
from pathlib import Path

# src 디렉토리를 sys.path에 추가
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import random
from dataclasses import fields

import torch

# internal
from environments.editing_env.offline_env import OfflineEditingEnv
from environments.editing_env.components.component import DocumentScore
from environments.editing_env.components.data import DocOfflineData
from methods.ppo.runner import PPORunner

from config.paths import LOGS_DIR, DATA_DIR
from utils.logger_factory import log

# 재현을 위한 seed 설정
SEED = 42

# ========== parameters for environment ==========
TERMINAL_THRESHOLD = 9.5  # 문서의 종합 품질 점수에 따라 종료할 한계점
REPEAT_PENALTY = 0.2  # 반복 액션에 대한 패널티 정도
# EDITOR_MODEL = "google/gemma-3-27b-it"  # 액션에 대한 LLM(or SLM)
EDITOR_MODEL = "qwen/qwen3-8b"  # 조금 더 성능이 좋지 않은 모델로 실험하기 위함

# 학습 시 LLM 비용에 대한 가중치로, COST_LAMBDA만큼 step마다 사용한 실제 비용에 곱하여 패널티 부과
# NOTE: 현재 LLM 비용 패널티는 고정해두었으니 튜닝하지 말 것
COST_LAMBDA = 1.0

STEP_PENLTY = 0.1  # step 하나 당 패널티 (ex) reward -= 2step * 패널티)

# JSONL_PATH = DATA_DIR / "paper_data" / "sequences_20251128_014521_tmp.jsonl"
JSONL_PATH = DATA_DIR / "paper_data" / "offline" / "sequences_20251128_014521_tmp.jsonl"

USE_SINGLE_SEQUENCE = True  # 오버피팅 모드 (첫 번째 시퀀스만 사용)
USE_LLM_JUDGE = False  # False면 rule-based evaluator 사용
USE_OFFLINE_REWARD = True  # offline_ppo.py 스타일 보상 함수 사용

# ========== parameters for train ==========
CHECKPOINT_DIR = None  # 학습 재개를 위한 설정 (저장된 체크포인트 디렉토리 경로)
SAVE_CHECKPOINT_DIR = LOGS_DIR / "checkpoints"
CHECKPOINT_INTERVAL = 1
LOG_INTERVAL = 100
TRAJECTORY_SAVE_INTERVAL = 1

NUM_EPISODES = 1000

# 재현을 위한 랜덤 시드 고정
random.seed(SEED)
torch.manual_seed(SEED)


def main():

    # load data
    log.info("데이터 로드")
    dataloader = DocOfflineData(jsonl_path=JSONL_PATH)

    env = OfflineEditingEnv(
        dataloader=dataloader,
        jsonl_path=JSONL_PATH,  # jsonl_path 명시적 전달
        max_steps=3,
        terminal_threshold=TERMINAL_THRESHOLD,  # 추가 (호환성)
        cost_lambda=COST_LAMBDA,
        repeat_penalty=REPEAT_PENALTY,  # 반복 패널티 감소
        editor_model=EDITOR_MODEL,  # 기존 설정 유지
        use_single_sequence=True,  # 오버피팅 모드 ON
    )

    # 평가 지표 (state)의 개수
    len_scores = len(fields(DocumentScore))
    runner = PPORunner(
        env=env,
        max_steps=3,
        state_dim=len_scores
        + 1
        + env.num_actions,  # 6가지 평가 기준 + step + last_action_one_hot
        num_actions=env.num_actions,
        gamma=0.95,
        lr=3e-4,
        clip_eps=0.2,
        K_epochs=4,
    )

    log.info("[학습 전] 정책:")
    runner.show_policy()

    # 학습
    rewards = runner.train(
        num_episodes=NUM_EPISODES,
        log_interval=LOG_INTERVAL,
        checkpoint_dir=SAVE_CHECKPOINT_DIR,
    )

    log.info("\n[학습 후] 정책:")
    runner.show_policy()

    # 평가
    runner.evaluate_greedy(max_steps=3)  # 평가 횟수 증가

    # 결과 요약
    log.info(f"\n{'='*60}")
    log.info("학습 결과 요약")
    log.info(f"{'='*60}")
    log.info(f"  초기 100ep 평균: {sum(rewards[:100])/100:+.3f}")
    log.info(f"  마지막 100ep 평균: {sum(rewards[-100:])/100:+.3f}")
    log.info(f"  개선도: {sum(rewards[-100:])/100 - sum(rewards[:100])/100:+.3f}")


if __name__ == "__main__":
    main()
