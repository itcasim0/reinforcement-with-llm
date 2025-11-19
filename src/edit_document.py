import sys
from pathlib import Path
# src 디렉토리를 sys.path에 추가
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import random

import torch

from dataloader.reconstruct_loader import ReconstructDataLoader
from environments.editing_env.env import EditingEnv
from methods.ppo import PPORunner
from utils.logger_factory import log


def main():

    # 재현을 위한 랜덤 시드 고정
    random.seed(42)
    torch.manual_seed(42)

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

    """
    TODO: num_episodes 증가 (학습 데이터 수 더 많게)
    TODO: max_steps 늘렸을 때 stop action 을 빨리 선택하는 경우가 있는지 확인 
    """
    log.info("학습 시작")
    runner.train(num_episodes=30)

    log.info("평가")
    runner.evaluate_greedy(max_steps=3)

    return


if __name__ == "__main__":
    main()
