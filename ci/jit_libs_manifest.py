#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Manifest + verify logic for the prebuilt AITER JIT ``.so`` release assets.

The 3 multi-GB JIT libs (``librmsnorm_fwd.so``, ``libmha_fwd.so``,
``libmha_bwd.so``) cost hours to build, so CI downloads them from a GitHub
release instead of rebuilding. This module owns the *contract* between the
producer (``ci/publish_jit_libs.sh``) and the consumer (``ci/fetch_jit_libs.sh``):

  * ``emit``   -- write a ``manifest.json`` describing a set of already-compressed
                  libs (sha256 of both the raw ``.so`` and the compressed blob).
  * ``verify`` -- given a downloaded ``manifest.json``, decide *skip-build* vs
                  *rebuild* by comparing the manifest's keys against the current
                  checkout (aiter submodule SHA + patch hash + GPU archs + ROCm).

The manifest is *keyed* by four inputs that fully determine ``.so`` validity:
``aiter_sha``, ``gpu_archs``, ``rocm_version`` and ``patch_hash``. ``verify``
treats ``aiter_sha`` + ``patch_hash`` (+ ``gpu_archs`` by default) as HARD gates
and ``rocm_version`` as advisory (the fetch host's ROCm can differ from the
build container's), unless ``--strict-rocm`` is given.

Everything here is import-safe + side-effect free so ``ci/test_jit_libs_manifest.py``
can unit-test the verify decision + a compress/decompress round-trip.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1

# The canonical 3 JIT libs (order = build order). Override with --libs.
DEFAULT_LIBS = ("librmsnorm_fwd.so", "libmha_fwd.so", "libmha_bwd.so")

# compression method -> (compressed-suffix, decompress argv-template). The
# template uses {src} and {dst} placeholders. Compression itself is done by
# ci/publish_jit_libs.sh (so the "prefer zstd -19 --long" recipe lives there);
# this table only needs the suffix for naming + the decompress command.
# NOTE: the zstd ``--long=27`` window (128 MiB) MUST match the compress side in
# ci/publish_jit_libs.sh so decompression never rejects a long-distance match.
COMPRESSORS = {
    "zstd": {"suffix": ".zst", "decompress": ["zstd", "-d", "-f", "--long=27", "-o", "{dst}", "{src}"]},
    "xz": {"suffix": ".xz", "decompress": ["xz", "-d", "-c", "{src}"]},
    "gzip": {"suffix": ".gz", "decompress": ["gzip", "-d", "-c", "{src}"]},
}


# --------------------------------------------------------------------------
# Hashing helpers
# --------------------------------------------------------------------------
def sha256_file(path: str | os.PathLike, chunk: int = 8 * 1024 * 1024) -> str:
    """Stream the file through sha256 (handles multi-GB libs without OOM)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def compute_patch_hash(repo_root: str | os.PathLike) -> str:
    """Hash repository-owned inputs that can change the JIT library bytes.

    ``patch_hash`` is the schema-v1 field name, but hashing every
    ``scripts/*.patch`` was wrong: editing the MaxText or XLA integration patch
    invalidated multi-GB MHA binaries even though those files never enter their
    build. Hash the JIT driver/config plus explicitly named ``aiter_jit_*.patch``
    files instead. AITER source itself is keyed separately by ``aiter_sha``.
    """
    root = Path(repo_root)
    inputs = [
        root / "Makefile",
        root / "jax_aiter" / "jit" / "build_jit.py",
        root / "jax_aiter" / "jit" / "optCompilerConfig.json",
        root / "ci" / "setup_jax.sh",
        root / "ci" / "jit_libs_manifest.py",
        root / "ci" / "publish_jit_libs.sh",
    ]
    common_dir = root / "csrc" / "common"
    if common_dir.is_dir():
        inputs.extend(path for path in common_dir.iterdir() if path.is_file())
    scripts_dir = root / "scripts"
    if scripts_dir.is_dir():
        inputs.extend(sorted(scripts_dir.glob("aiter_jit_*.patch")))
    inputs = [path for path in inputs if path.is_file()]
    if not inputs:
        return "none"

    h = hashlib.sha256()
    for path in sorted(inputs):
        # Include a stable repository-relative path so a rename changes the
        # hash without making it depend on the checkout's absolute location.
        h.update(path.relative_to(root).as_posix().encode("utf-8"))
        h.update(b"\0")
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return "sha256:" + h.hexdigest()


def compute_aiter_sha(repo_root: str | os.PathLike) -> str:
    """Resolve the consumed AITER commit, even before submodule checkout."""
    root = Path(repo_root)
    aiter_dir = root / "third_party" / "aiter"
    try:
        out = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={aiter_dir.resolve()}",
                "-C",
                str(aiter_dir),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        pass
    try:
        out = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root.resolve()}",
                "-C",
                str(root),
                "ls-tree",
                "HEAD",
                "third_party/aiter",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        fields = out.stdout.split()
        if len(fields) >= 3 and fields[1] == "commit":
            return fields[2]
    except Exception:
        pass
    return "unknown"


def detect_rocm_version() -> str:
    """Best-effort ROCm version. ``ROCM_VERSION`` env wins, then the version
    file, then ``hipcc --version``, else ``"unknown"``.

    NOTE: pass ``ROCM_VERSION`` explicitly when the tool runs outside the build
    container (e.g. compressing on a host whose ``/opt/rocm`` differs from the
    container the libs were built in).
    """
    env = os.environ.get("ROCM_VERSION", "").strip()
    if env:
        return env
    for vf in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        try:
            txt = Path(vf).read_text().strip()
            if txt:
                # Strip a trailing "-<build>" suffix (e.g. "7.2.0-12345").
                return txt.split("-", 1)[0]
        except Exception:
            pass
    try:
        out = subprocess.run(["hipcc", "--version"], capture_output=True, text=True, check=True)
        for line in out.stdout.splitlines():
            if "HIP version" in line:
                return line.split(":", 1)[1].strip().split("-", 1)[0]
    except Exception:
        pass
    return "unknown"


def normalize_archs(archs: str | None) -> list[str]:
    """Split a ``gfx942;gfx950`` / ``gfx942,gfx950`` string into a sorted list."""
    if not archs:
        return []
    out: list[str] = []
    for tok in archs.replace(",", ";").replace(" ", ";").split(";"):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return sorted(set(out))


def compute_cache_id(
    *, aiter_sha: str, gpu_archs: str, patch_hash: str, rocm_version: str
) -> str:
    """Stable identifier for one immutable set of JIT build inputs."""
    payload = "\0".join(
        (
            aiter_sha,
            ";".join(normalize_archs(gpu_archs)),
            patch_hash,
            rocm_version,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def compute_current_cache_id(
    repo_root: str | os.PathLike, gpu_archs: str, rocm_version: str
) -> str:
    root = Path(repo_root)
    return compute_cache_id(
        aiter_sha=compute_aiter_sha(root),
        gpu_archs=gpu_archs,
        patch_hash=compute_patch_hash(root),
        rocm_version=rocm_version,
    )


# --------------------------------------------------------------------------
# Manifest build / load
# --------------------------------------------------------------------------
def build_manifest(
    *,
    aiter_build_dir: str | os.PathLike,
    dist_dir: str | os.PathLike,
    compression: str,
    tag: str,
    aiter_sha: str,
    gpu_archs: str,
    rocm_version: str,
    patch_hash: str,
    libs: tuple[str, ...] = DEFAULT_LIBS,
) -> dict:
    """Build the manifest dict from the raw libs + their compressed blobs.

    For each lib ``<name>`` it expects ``<dist_dir>/<name><suffix>`` to already
    exist (compressed by the publish script). Records sha256 + size of BOTH the
    raw ``.so`` and the compressed blob.
    """
    if compression not in COMPRESSORS:
        raise ValueError(f"unknown compression '{compression}' (have {list(COMPRESSORS)})")
    suffix = COMPRESSORS[compression]["suffix"]
    aiter_build_dir = Path(aiter_build_dir)
    dist_dir = Path(dist_dir)

    files = []
    for name in libs:
        raw = aiter_build_dir / name
        comp = dist_dir / (name + suffix)
        if not raw.is_file():
            raise FileNotFoundError(f"raw lib missing: {raw}")
        if not comp.is_file():
            raise FileNotFoundError(f"compressed blob missing: {comp}")
        files.append({
            "name": name,
            "size": raw.stat().st_size,
            "sha256": sha256_file(raw),
            "compressed_name": comp.name,
            "compressed_size": comp.stat().st_size,
            "compressed_sha256": sha256_file(comp),
        })

    return {
        "schema": SCHEMA_VERSION,
        "tag": tag,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aiter_sha": aiter_sha,
        "gpu_archs": gpu_archs,
        "rocm_version": rocm_version,
        "patch_hash": patch_hash,
        "compression": compression,
        "files": files,
    }


def load_manifest(path: str | os.PathLike) -> dict:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Verify (the skip-build vs rebuild decision)
# --------------------------------------------------------------------------
def verify_manifest(
    manifest: dict,
    *,
    current_aiter_sha: str,
    current_patch_hash: str,
    expected_gpu_archs: str | None = None,
    current_rocm_version: str | None = None,
    strict_gpu_archs: bool = True,
    strict_rocm: bool = False,
) -> tuple[bool, str, list[str]]:
    """Decide *skip-build* (ok=True) vs *rebuild* (ok=False).

    HARD gates (mismatch -> rebuild):
      * ``aiter_sha``
      * ``patch_hash``
      * ``gpu_archs``    (when ``strict_gpu_archs``; default True)

    ADVISORY (warn only unless ``strict_rocm``):
      * ``rocm_version`` -- the fetch host's ROCm frequently differs from the
        build container's, so a mismatch is reported but does not force a
        rebuild by default.

    Returns ``(ok, decision, reasons)`` where ``decision`` is ``"skip-build"``
    or ``"rebuild"`` and ``reasons`` explains every comparison.
    """
    reasons: list[str] = []
    ok = True

    m_aiter = manifest.get("aiter_sha", "")
    if m_aiter == current_aiter_sha and current_aiter_sha not in ("", "unknown"):
        reasons.append(f"OK   aiter_sha matches ({current_aiter_sha[:12]})")
    else:
        ok = False
        reasons.append(f"FAIL aiter_sha mismatch (manifest={m_aiter[:12] or '?'} current={current_aiter_sha[:12] or '?'})")

    m_patch = manifest.get("patch_hash", "")
    if m_patch == current_patch_hash:
        reasons.append(f"OK   patch_hash matches ({_short_hash(current_patch_hash)})")
    else:
        ok = False
        reasons.append(f"FAIL patch_hash mismatch (manifest={_short_hash(m_patch)} current={_short_hash(current_patch_hash)})")

    if expected_gpu_archs is not None:
        m_archs = normalize_archs(manifest.get("gpu_archs"))
        e_archs = normalize_archs(expected_gpu_archs)
        if m_archs == e_archs:
            reasons.append(f"OK   gpu_archs matches ({';'.join(e_archs)})")
        elif strict_gpu_archs:
            ok = False
            reasons.append(f"FAIL gpu_archs mismatch (manifest={';'.join(m_archs)} expected={';'.join(e_archs)})")
        else:
            reasons.append(f"WARN gpu_archs differ (manifest={';'.join(m_archs)} expected={';'.join(e_archs)}) [advisory]")

    if current_rocm_version is not None:
        m_rocm = manifest.get("rocm_version", "")
        if m_rocm == current_rocm_version:
            reasons.append(f"OK   rocm_version matches ({current_rocm_version})")
        elif strict_rocm:
            ok = False
            reasons.append(f"FAIL rocm_version mismatch (manifest={m_rocm} current={current_rocm_version})")
        else:
            reasons.append(f"WARN rocm_version differs (manifest={m_rocm} current={current_rocm_version}) [advisory]")

    return ok, ("skip-build" if ok else "rebuild"), reasons


def _short_hash(h: str) -> str:
    if not h or h == "none":
        return h or "?"
    return h[:18] + "..." if len(h) > 18 else h


# --------------------------------------------------------------------------
# Compression round-trip helpers (used by fetch + the unit test)
# --------------------------------------------------------------------------
def decompress_file(src: str | os.PathLike, dst: str | os.PathLike, compression: str) -> None:
    """Decompress ``src`` -> ``dst`` using the tool for ``compression``.

    Raises ``RuntimeError`` if the tool is missing or fails (callers map this to
    a rebuild signal rather than a hard crash).
    """
    if compression not in COMPRESSORS:
        raise RuntimeError(f"unknown compression '{compression}'")
    spec = COMPRESSORS[compression]
    tool = spec["decompress"][0]
    if shutil.which(tool) is None:
        raise RuntimeError(f"decompressor '{tool}' not found on PATH")
    argv = [a.format(src=str(src), dst=str(dst)) for a in spec["decompress"]]
    # zstd writes to {dst} via -o; xz/gzip stream to stdout -> redirect to dst.
    if "{dst}" in " ".join(spec["decompress"]):
        subprocess.run(argv, check=True)
    else:
        with open(dst, "wb") as out:
            subprocess.run(argv, check=True, stdout=out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _cmd_cache_id(args: argparse.Namespace) -> int:
    rocm_version = args.rocm_version or detect_rocm_version()
    cache_id = compute_current_cache_id(
        args.repo_root, args.gpu_archs, rocm_version
    )
    print(f"{args.prefix}-{cache_id}" if args.prefix else cache_id)
    return 0


def _cmd_emit(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    aiter_sha = args.aiter_sha or compute_aiter_sha(repo_root)
    patch_hash = args.patch_hash or compute_patch_hash(repo_root)
    rocm_version = args.rocm_version or detect_rocm_version()
    libs = tuple(s.strip() for s in args.libs.split(",")) if args.libs else DEFAULT_LIBS

    manifest = build_manifest(
        aiter_build_dir=args.aiter_build_dir,
        dist_dir=args.dist_dir,
        compression=args.compression,
        tag=args.tag,
        aiter_sha=aiter_sha,
        gpu_archs=args.gpu_archs,
        rocm_version=rocm_version,
        patch_hash=patch_hash,
        libs=libs,
    )
    out = args.out or os.path.join(args.dist_dir, "manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"[manifest] wrote {out}")
    print(json.dumps(manifest, indent=2))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    manifest = load_manifest(args.manifest)
    current_aiter_sha = args.aiter_sha or compute_aiter_sha(repo_root)
    current_patch_hash = args.patch_hash or compute_patch_hash(repo_root)
    current_rocm = args.rocm_version or detect_rocm_version()

    ok, decision, reasons = verify_manifest(
        manifest,
        current_aiter_sha=current_aiter_sha,
        current_patch_hash=current_patch_hash,
        expected_gpu_archs=args.gpu_archs,
        current_rocm_version=current_rocm,
        strict_gpu_archs=not args.no_strict_gpu_archs,
        strict_rocm=args.strict_rocm,
    )
    for r in reasons:
        print(f"[verify] {r}")
    print(f"[verify] decision={decision}")
    # Optionally emit a GITHUB_OUTPUT line for the workflow to consume.
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"decision={decision}\n")
            f.write(f"rebuild={'false' if ok else 'true'}\n")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AITER JIT-lib release manifest tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache-id", help="print immutable JIT input identifier")
    c.add_argument("--repo-root", default=".")
    c.add_argument("--gpu-archs", required=True)
    c.add_argument("--rocm-version", default="")
    c.add_argument("--prefix", default="")
    c.set_defaults(func=_cmd_cache_id)

    e = sub.add_parser("emit", help="write manifest.json for compressed libs")
    e.add_argument("--repo-root", default=".")
    e.add_argument("--aiter-build-dir", default="build/aiter_build")
    e.add_argument("--dist-dir", required=True, help="dir holding the compressed blobs")
    e.add_argument("--compression", required=True, choices=list(COMPRESSORS))
    e.add_argument("--tag", required=True)
    e.add_argument("--gpu-archs", required=True)
    e.add_argument("--rocm-version", default="")
    e.add_argument("--aiter-sha", default="")
    e.add_argument("--patch-hash", default="")
    e.add_argument("--libs", default="", help="comma list (default: the 3 JIT libs)")
    e.add_argument("--out", default="")
    e.set_defaults(func=_cmd_emit)

    v = sub.add_parser("verify", help="decide skip-build vs rebuild")
    v.add_argument("--repo-root", default=".")
    v.add_argument("--manifest", required=True)
    v.add_argument("--gpu-archs", default=None)
    v.add_argument("--rocm-version", default="")
    v.add_argument("--aiter-sha", default="")
    v.add_argument("--patch-hash", default="")
    v.add_argument("--no-strict-gpu-archs", action="store_true")
    v.add_argument("--strict-rocm", action="store_true")
    v.set_defaults(func=_cmd_verify)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
