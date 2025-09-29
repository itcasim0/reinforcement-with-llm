from pathlib import Path
import shutil

from datetime import datetime
from zoneinfo import ZoneInfo


def today_datetime() -> str:
    """Get current datetime string in 'Asia/Seoul' timezone.

    Returns:
        str: Current datetime in the format 'YYYYMMDDTHHMMSS'.
    """
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%dT%H%M%S")


def mkdir(d: str | Path, exist_backup: bool = True) -> None:
    """Create a directory, backing up if it already exists.

    Args:
        d (str | Path): Directory path to create.
        exist_backup (bool, optional): Whether to back up the existing directory. Defaults to True.
    """
    d = Path(d)
    if exist_backup:
        backup(d)
    d.mkdir(parents=True, exist_ok=True)


def backup(p: str | Path) -> None:
    """Back up a file or directory by renaming it with a timestamp.

    If the path is a file, it renames the file with a timestamp.
    If the path is a directory, it moves the directory to a new name with a timestamp.

    Args:
        p (str | Path): Path to the file or directory to back up.
    """
    p = Path(p)
    if p.is_file():
        file_name = p.stem
        ext = p.suffix
        p.rename(p.parent / f"{file_name}_{today_datetime()}{ext}")
    elif p.is_dir():
        shutil.move(p, str(p) + f"_{today_datetime()}")