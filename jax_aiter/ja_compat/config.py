import os
from pathlib import Path


def get_repo_root() -> Path:
    # Prefer explicit env; fall back to common repo layout.
    root = os.environ.get("JA_ROOT_DIR")
    if root:
        return Path(root).resolve()
    return Path(__file__).resolve().parents[3]


def get_configs_dir() -> Path:
    # matches aiter's "aiter/configs/" dir.
    return get_repo_root() / "build" / "hipifiedaiter" / "configs"


def get_shared_lib() -> Path:
    # The .so location of JA.
    return get_repo_root() / "build" / "bin" / "libjax_aiter.so"
