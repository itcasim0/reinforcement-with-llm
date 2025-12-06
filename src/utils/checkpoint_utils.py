from pathlib import Path


def get_latest_checkpoint(base_dir: str = "logs/checkpoints") -> Path:
    """가장 최근 체크포인트 찾기"""
    checkpoints_path = Path(base_dir)

    if not checkpoints_path.exists():
        raise FileNotFoundError(f"체크포인트 디렉토리가 없습니다: {checkpoints_path}")

    checkpoint_dirs = [
        d
        for d in checkpoints_path.iterdir()
        if d.is_dir() and len(d.name) == 15 and "T" in d.name
    ]

    if not checkpoint_dirs:
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoints_path}")

    return sorted(checkpoint_dirs, key=lambda x: x.name, reverse=True)[0]
