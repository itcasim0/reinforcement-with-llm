"""
edit_document_offline.py에는 사전에 미리 LLM을 통해 도출된 결과를 가지고 학습과 평가하는 코드가 공존하므로 참고
"""

import sys
import os
from pathlib import Path

# src 디렉토리를 sys.path에 추가
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import random

import torch

# internal
from environments.editing_env.env import OfflineEditingEnv, StrictEvaluator
from methods.ppo import PPORunner

from config.paths import LOGS_DIR, DATA_DIR
from utils.logger_factory import log

# 재현을 위한 seed 설정
SEED = 42

# parameters for environment
TERMINAL_THRESHOLD = 9.5  # 문서의 종합 품질 점수에 따라 종료할 한계점
REPEAT_PENALTY = 0.2  # 반복 액션에 대한 패널티 정도
# EDITOR_MODEL = "google/gemma-3-27b-it"  # 액션에 대한 LLM(or SLM)
EDITOR_MODEL = "qwen/qwen3-8b"  # 조금 더 성능이 좋지 않은 모델로 실험하기 위함

# parameters for offline environment (offline_ppo.py와 env.py의 OfflineEditingEnv에 맞춤)
# offline_ppo.py와 동일하게 스크립트 디렉토리 기준 경로 사용
script_dir = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH = os.path.join(
    script_dir, "first_doc_all_sequences_prefix_reuse_with_noise.jsonl"
)
USE_SINGLE_SEQUENCE = True  # 오버피팅 모드 (첫 번째 시퀀스만 사용)
USE_LLM_JUDGE = False  # False면 rule-based evaluator 사용
USE_OFFLINE_REWARD = True  # offline_ppo.py 스타일 보상 함수 사용

# parameters for train
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
    # 데이터 파일 경로
    jsonl_path = (
        DATA_DIR
        / "paper_data"
        / "first_doc_all_sequences_prefix_reuse_with_noise.jsonl"
    )

    if not os.path.exists(jsonl_path):
        log.info(f"[ERROR] 데이터 파일을 찾을 수 없습니다: {jsonl_path}")
        exit(1)

    # === 먼저 평가기 테스트 ===
    log.info("=" * 60)
    log.info("깐깐한 평가기 테스트")
    log.info("=" * 60)

    # StrictEvaluator 임포트 (env.py에 추가했으므로 사용 가능)

    evaluator = StrictEvaluator()

    # 데이터 로드해서 base vs final 비교
    import json

    with open(jsonl_path, "r", encoding="utf-8") as f:
        seq = json.loads(f.readline())
    
    base_text = seq["base_text"]

    # 문제점 분석
    issues = evaluator.detailed_report(base_text)
    log.info(f"\n[저품질 초록의 문제점]")
    log.info(f"  모호한 표현: {issues['vague'][:5]}...")
    log.info(f"  어색한 어미: {issues['awkward'][:5]}...")
    log.info(f"  구어체: {issues['colloquial'][:5]}...")

    # === 환경 및 학습 ===
    log.info("\n" + "=" * 60)
    log.info("RL 학습 시작")
    log.info("=" * 60)

    env = OfflineEditingEnv(
        jsonl_path=jsonl_path,  # jsonl_path 명시적 전달
        documents=[],  # 오프라인 모드라 빈 리스트
        max_steps=3,
        terminal_threshold=TERMINAL_THRESHOLD,  # 추가 (호환성)
        cost_lambda=0.5,  # 비용 패널티 감소 (학습 용이)
        repeat_penalty=REPEAT_PENALTY,  # 반복 패널티 감소
        editor_model=EDITOR_MODEL,  # 기존 설정 유지
        use_single_sequence=True,  # 오버피팅 모드 ON
        use_llm_judge=False,  # StrictEvaluator 사용
        use_offline_reward=True,  # 오프라인 보상 사용
    )

    runner = PPORunner(
        env=env,
        max_steps=3,
        state_dim=4 + 1 + env.num_actions,  # g,r,c,o + step + last_action_one_hot
        num_actions=env.num_actions,
        gamma=0.95,
        lr=3e-4,
        clip_eps=0.2,
        K_epochs=4,
        # hidden_size=128,                    # offline_ppo.py와 동일
    )

    log.info("\n[학습 전] 정책:")
    runner.show_policy()

    # 학습
    rewards = runner.train(num_episodes=NUM_EPISODES, log_interval=LOG_INTERVAL)

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
