from pathlib import Path

# 프로젝트 루트 디렉토리 설정
# src/config/paths.py 위치 기준으로 루트 디렉토리 계산
# parent -> config
# parent.parent -> src
# parent.parent.parent -> project root (reinforcement-with-llm)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
SRC_DIR = ROOT_DIR / "src"