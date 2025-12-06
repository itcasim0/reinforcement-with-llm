"""
edit_document.py에는 학습과 평가하는 코드가 공존하므로 참고
"""

import sys
from pathlib import Path
import random
from dataclasses import fields

import torch

# 프로젝트 루트를 Python 경로에 추가
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(1, str(root_dir / "src"))

# internal
from src.dataloader.reconstruct_loader import DomesticReconstructDataLoader
from src.environments.editing_env.base_env import EditingEnv
from src.environments.editing_env.components.component import DocumentScore
from src.methods.ppo.runner import PPORunner

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
SAVE_CHECKPOINT_DIR = LOGS_DIR / "checkpoints" / "ppo"
CHECKPOINT_INTERVAL = 10
LOG_INTERVAL = 10
TRAJECTORY_SAVE_INTERVAL = 1

BUFFER_SIZE = 32  # 학습 전에 모을 step 수
BATCH_SIZE = 16  # 미니배치 크기
K_EPOCHS = 2  # BUFFER_SIZE만큼 쌓인 후 update하는 횟수

NUM_EPISODES = 1000

# estimator에서 사용하는 값
GAMMA = 0.95
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.03
CLIP_EPS = 0.2


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
        lr=3e-4,
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
    )

    # 평가 시작
    log.info("평가")
    runner.evaluate_greedy()
    return


if __name__ == "__main__":
    main()
