import sys
from pathlib import Path

# src 디렉토리를 sys.path에 추가
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import random
import argparse

import torch

from dataloader.reconstruct_loader import ReconstructDataLoader
from environments.editing_env.env import EditingEnv
from methods.ppo import PPORunner

from config.paths import LOGS_DIR
from utils.logger_factory import log

SEED = 42

# 재현을 위한 랜덤 시드 고정
random.seed(SEED)
torch.manual_seed(SEED)


def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="PPO 기반 문서 교정 강화학습")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="체크포인트 디렉토리 또는 파일 경로 (학습 재개 시 사용)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=LOGS_DIR / "checkpoints",
        help="체크포인트 저장 디렉토리 (기본값: ./logs/checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="체크포인트 저장 주기 (에피소드 단위, 기본값: 1)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=30,
        help="학습할 에피소드 수 (기본값: 30)",
    )
    args = parser.parse_args()

    # load data
    log.info("데이터 로드")
    dataloader = ReconstructDataLoader()
    documents = dataloader.get_reconstructed_text(max_docs=5)

    # 강화학습 환경 구성
    log.info("강화학습 환경 구성")
    env = EditingEnv(
        documents=documents,
        max_steps=3,
        terminal_threshold=4.5,
        cost_lambda=1.0,
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
        K_epochs=4,
    )

    # 체크포인트에서 재개
    if args.resume:
        try:
            runner.load_checkpoint(args.resume)
            log.info(f"체크포인트에서 학습 재개: {args.resume}")
        except FileNotFoundError as e:
            log.error(f"체크포인트 로드 실패: {e}")
            return

    # 학습 시작
    log.info("학습 시작")
    runner.train(
        num_episodes=args.num_episodes,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )

    # 평가 시작
    log.info("평가")
    runner.evaluate_greedy(max_steps=3)

    return


if __name__ == "__main__":
    main()
