import hashlib
import os
import sys
from pathlib import Path
from typing import List


def get_data_dir() -> Path:
    env_override = os.environ.get("RPG_DATA_DIR", "").strip()
    if env_override:
        p = Path(env_override)
        p.mkdir(parents=True, exist_ok=True)
        return p

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))

    p = base / "rpg-agents"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_db_path() -> Path:
    override = os.environ.get("DB_PATH", "").strip()
    if override:
        return Path(override)
    return get_data_dir() / "agents.db"


def configure_console():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def hp_bar(current: int, maximum: int, width: int = 20) -> str:
    if maximum <= 0:
        return "[" + "-" * width + "]"
    filled = int((current / maximum) * width)
    filled = max(0, min(width, filled))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{maximum}"


def deterministic_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest(), 16)


def embed_text(text: str) -> List[float]:
    words = text.lower().split()
    vec = [0.0] * 64
    for i, word in enumerate(words[:64]):
        h = deterministic_hash(word)
        idx = h % 64
        vec[idx] += 1.0 / (i + 1)
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, min(1.0, dot))
