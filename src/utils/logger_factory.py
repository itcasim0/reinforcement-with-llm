import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

DEFAULT_LOG_FORMAT = "%(asctime)s #%(process)d \t %(filename)s(%(lineno)s) \t %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_PATH = "../logs/py.log"


def set_stream_handler(
    logger: logging.Logger,
    stream_format: str = DEFAULT_LOG_FORMAT,
):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(stream_format))
    logger.addHandler(stream_handler)

    return None


def _mkdir_log_path(log_path: str | Path):
    if not log_path:
        log_path = DEFAULT_LOG_PATH

    log_path = Path(log_path)
    # 경로상에 dir가 없을 경우, dir 생성
    log_path.parent.mkdir(parents=True, exist_ok=True)

    return log_path


def set_file_handler(
    logger: logging.Logger,
    log_path: str | Path = None,
    log_format: str = DEFAULT_LOG_FORMAT,
):
    log_path = _mkdir_log_path(log_path)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    return None


def set_timed_rotating_file_handler(
    logger: logging.Logger,
    log_path: str | Path,
    log_format: str = DEFAULT_LOG_FORMAT,
    when: str = "h",
    interval: int = 1,
    backup_count: int = 0,
):
    log_path = _mkdir_log_path(log_path)

    file_handler = TimedRotatingFileHandler(
        log_path, when=when, interval=interval, backupCount=backup_count
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    return None


def _global_logger() -> logging.Logger:
    logger = logging.getLogger("main")
    logger.setLevel(DEFAULT_LOG_LEVEL)

    set_stream_handler(logger, DEFAULT_LOG_FORMAT)

    set_timed_rotating_file_handler(
        logger, DEFAULT_LOG_PATH, DEFAULT_LOG_FORMAT, "W0", 1
    )

    return logger


log = _global_logger()
