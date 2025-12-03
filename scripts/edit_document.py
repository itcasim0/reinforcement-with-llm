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

# 입력 데이터셋 json 파일
INPUT_DATA = DATA_DIR / "paper_data" / "reconstruct"

# ========== parameters for environment ==========
TERMINAL_THRESHOLD = 9.5  # 문서의 종합 품질 점수에 따라 종료할 한계점
REAPEAT_PANELTY = 0.5  # 반복 액션에 대한 패널티 정도
# EDITOR_MODEL = "google/gemma-3-27b-it"  # 액션에 대한 LLM(or SLM)
EDITOR_MODEL = "qwen/qwen3-8b"  # 조금 더 성능이 좋지 않은 모델로 실험하기 위함

# 학습 시 LLM 비용에 대한 가중치로, COST_LAMBDA만큼 step마다 사용한 실제 비용에 곱하여 패널티 부과
# NOTE: 현재 LLM 비용 패널티는 고정해두었으니 튜닝하지 말 것
COST_LAMBDA = 1.0

STEP_PENLTY = 0.1  # step 하나 당 패널티 (ex) reward -= 2step * 패널티)


# ========== parameters for train ==========
CHECKPOINT_DIR = None  # 학습 재개를 위한 설정 (저장된 체크포인트 디렉토리 경로)
SAVE_CHECKPOINT_DIR = LOGS_DIR / "checkpoints"
CHECKPOINT_INTERVAL = 1
LOG_INTERVAL = 1
TRAJECTORY_SAVE_INTERVAL = 1

NUM_EPISODES = 10

# 재현을 위한 랜덤 시드 고정
random.seed(SEED)
torch.manual_seed(SEED)


def main():

    # load data
    log.info("데이터 로드")
    dataloader = DomesticReconstructDataLoader(json_path=INPUT_DATA)

    # 강화학습 환경 구성
    log.info("강화학습 환경 구성")
    env = EditingEnv(
        dataloader=dataloader,
        max_steps=3,
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
        max_steps=3,
        state_dim=len_scores
        + 1
        + env.num_actions,  # g,r,c,o + step + last_action_one_hot
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
    )

    # 평가 시작
    log.info("평가")
    runner.evaluate_greedy()

    return


if __name__ == "__main__":
    main()
