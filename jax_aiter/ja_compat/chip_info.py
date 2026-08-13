# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# (Ruturaj4): Move this later to jit/utils once we add jit functionality.
import os
import shutil
import functools
import subprocess


def executable_path(executable: str) -> str:
    """
    Return the path to the executable.

    Args:
        executable (str): The name of the executable.

    Returns:
        The path to the executable.
    """
    path = shutil.which(executable)
    if not path:
        home = _find_rocm_home()
        if home:
            path = shutil.which(os.path.join(home, "bin", executable))
        assert (
            path is not None
        ), f"Could not find {executable} in PATH or ROCM_HOME({home})"
    return os.path.realpath(path)


@functools.lru_cache(maxsize=1)
def get_gfx():
    gfx = os.getenv("GPU_ARCHS", "native")
    if gfx == "native":
        try:
            rocminfo = executable_path("rocminfo")
            result = subprocess.run(
                [rocminfo], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            output = result.stdout
            for line in output.split("\n"):
                if "gfx" in line.lower():
                    return line.split(":")[-1].strip()
        except Exception as e:
            raise RuntimeError(f"Get GPU arch from rocminfo failed {str(e)}")
    elif ";" in gfx:
        gfx = gfx.split(";")[-1]
    return gfx
