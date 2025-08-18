# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
import csv, os, logging
from pathlib import Path
from functools import lru_cache
from typing import Dict, Tuple, Any, Optional
import pandas as pd

from .dtypes import str2bool

log = logging.getLogger("aiter.compat")
Key = Tuple[int, int, int, bool, str]


def _cfg_dir() -> Path:
    root = os.environ.get("AITER_ROOT_DIR")
    return (
        (Path(root) / "aiter" / "configs")
        if root
        else (
            Path(__file__).resolve().parents[2]
            / "third_party"
            / "aiter"
            / "aiter"
            / "configs"
        )
    )


@lru_cache(maxsize=256)
def _load_table(tuned_file: str) -> Dict[Key, Dict[str, Any]]:
    path = _cfg_dir() / tuned_file
    if not path.is_file():
        raise FileNotFoundError(f"Tuned file not found: {path}")

    # if not hasattr(get_ASMGEMM_config, "asmgemm_dict"):
    df = pd.read_csv(path).drop_duplicates()

    if "bias" in df.columns:
        df["bias"] = df["bias"].apply(str2bool)
    if "outdtype" in df.columns:
        df["outdtype"] = df["outdtype"].apply(lambda s: s.split(".")[-1])

    for col in ("M", "N", "K"):
        if col in df.columns:
            df[col] = df[col].astype(int)

    idx_cols = ["M", "N", "K", "bias", "outdtype"]
    table = df.set_index(idx_cols).to_dict("index")

    # Include M, N, K, bias back in each entry.
    for key, val in table.items():
        val.update({"M": key[0], "N": key[1], "K": key[2], "bias": key[3]})

    return table


# Cached inner function that only takes hashable primitives.
@lru_cache(maxsize=1024)
def _get_ASM_cfg_cached(
    M: int, N: int, K: int, bias: bool, outdtype, tuned_file: str
) -> Optional[Dict[str, Any]]:
    cfg = _load_table(tuned_file).get((M, N, K, bias, outdtype.dtype.name))
    if cfg is not None:
        log.info(f"shape M:{M}, N:{N}, K:{K} is tuned, in ASMGEMM !")
    return cfg


# Public API (no cache): normalize to primitives first.
def get_ASMGEMM_config(
    M: int, N: int, K: int, bias: bool, outdtype, tuned_file: str = "asm_a8w8_gemm.csv"
) -> Optional[Dict[str, Any]]:
    return _get_ASM_cfg_cached(
        int(M), int(N), int(K), bool(bias), outdtype, str(tuned_file)
    )
