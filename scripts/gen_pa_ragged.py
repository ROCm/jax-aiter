#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Render aiter's paged-attention kernels as ordinary sources, for AOT compile.

This replaces ``scripts/prebuild_pa_ragged.py``, which drove aiter's own Python
JIT: it compiled each configuration into ``$HOME/.aiter/build/<md5>/lib.so`` and
``paged_attention_ja.so`` ``dlopen``ed it at run time. That design came from
aiter's packaging convention, not from anything the kernel needs, and it cost us
a CI prebuild step, a runtime cache directory outside the wheel, and a jinja2
install -- while the generated library, having no ``RUNPATH``, could not even be
loaded in a container that had not been hand-taught where ROCm lives.

The generated source turns out to contain no kernel logic at all: it is one
``extern "C"`` function instantiating two C++ templates from ``pa_ragged.cuh``
with literal constants. Jinja is standing in for template arguments. So the
kernels compile straight into the shim like ``append_kv`` and ``paged_prefill``
already do, and both the cache and the ``dlopen`` disappear.

What this emits into ``--out-dir``:
  <func_name>.cpp            one per configuration, rendered from aiter's template
  pa_dispatch_generated.h    the ``extern "C"`` declarations plus an X-macro list

``Makefile.kv`` compiles the sources and ``csrc/ffi/kv/paged_attention_ja.cu``
expands the X-macro into a static name -> function table.

Two things stay deliberately unchanged from the JIT design, to keep this a
mechanism swap rather than a redesign: configurations are still keyed by the md5
name ``jax_aiter.kv.pa_config.func_name`` computes, and the FFI still receives
that name as an attribute. Only the resolution changes.

Rendering aiter's own template, rather than hand-writing the wrapper, is what
keeps this honest across the nightly aiter pin bump: the kernel launch arguments
stay owned by aiter. ``--check-signature`` guards the one thing that would break
silently -- a drift between the template's ``extern "C"`` block and the function
pointer type in ``paged_attention_ja.cu``.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from jax_aiter.kv.pa_config import (  # noqa: E402
    DTYPE_BF16,
    DTYPE_FP16,
    PARTITION_SIZE,
    func_name as config_func_name,
)

# sha256 of the template's normalised `extern "C"` block. The 24-argument
# signature it declares is transcribed by hand into PagedAttentionFn in
# csrc/ffi/kv/paged_attention_ja.cu; nothing but this check would notice if an
# aiter bump reordered or retyped an argument, and the failure mode is a silent
# ABI mismatch -- scalars landing in pointer slots -- not a compile error.
# To update: read the new block, fix PagedAttentionFn to match, then paste the
# hash this script prints.
EXPECTED_SIGNATURE_SHA256 = (
    "bdc5b2a46f0200370464c8687e37b709032545628535cc0f6ee2de4f2e9bbfab"
)

# The C signature the shim's PagedAttentionFn mirrors, in declaration order.
_DECL_PARAMS = (
    "void *, void *, void *, void *, void *, "
    "int *, int *, int *, "
    "const float *, const float *, const float *, const float *, const float *, "
    "float, float, "
    "const int, const int, const int, const int, const int, const int, const int, "
    "const int, void *"
)


def template_path(aiter_dir: Path) -> Path:
    return aiter_dir / "csrc" / "cpp_itfs" / "pa" / "pa_ragged.cpp.jinja"


def signature_sha256(template_text: str) -> str:
    """Hash the template's ``extern "C"`` block, ignoring whitespace and the name.

    The block is reproduced per configuration with a different ``func_name``, so
    the name is replaced by a placeholder before hashing; only the argument list
    is being pinned.
    """
    match = re.search(r'extern\s+"C"\s*\{(.*?)\}', template_text, re.DOTALL)
    if match is None:
        raise RuntimeError(
            f"no `extern \"C\"` block in the template. aiter has restructured "
            f"pa_ragged.cpp.jinja; re-derive PagedAttentionFn in "
            f"csrc/ffi/kv/paged_attention_ja.cu before trusting this build."
        )
    block = match.group(1)
    block = block.replace("{{func_name}}", "FUNC")
    block = re.sub(r"\s+", " ", block).strip()
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def check_signature(template_text: str) -> None:
    actual = signature_sha256(template_text)
    if actual != EXPECTED_SIGNATURE_SHA256:
        raise SystemExit(
            "pa_ragged.cpp.jinja's `extern \"C\"` signature has changed.\n"
            f"  expected {EXPECTED_SIGNATURE_SHA256}\n"
            f"  actual   {actual}\n"
            "The kernel entry is called through a hand-transcribed function\n"
            "pointer type (PagedAttentionFn in csrc/ffi/kv/paged_attention_ja.cu).\n"
            "A reordered or retyped argument would not fail to compile -- it\n"
            "would put scalars in pointer slots at run time. Reconcile the two,\n"
            "then set EXPECTED_SIGNATURE_SHA256 in this file to the actual hash."
        )


def default_configs(head_size: int, block_size: int, npar_loops: int) -> list[dict]:
    """The set the M2 tests exercise: bf16 and fp16, MHA/GQA/MQA ratios.

    Identical to what scripts/prebuild_pa_ragged.py built, so this is a pure
    mechanism change. Widening it is the supported way to add a configuration --
    the same contract BP_FILTER documents for paged prefill.
    """
    configs = []
    for dtype in (DTYPE_BF16, DTYPE_FP16):
        for gqa_ratio in (1, 4, 8):
            configs.append(
                {
                    "gqa_ratio": gqa_ratio,
                    "head_size": head_size,
                    "npar_loops": npar_loops,
                    "dtype": dtype,
                    "kv_dtype": dtype,
                    "kv_cache_dtype": "auto",
                    "out_dtype": dtype,
                    "block_size": block_size,
                    "alibi_enabled": "false",
                }
            )
    return configs


def render(template_text: str, cfg: dict, name: str) -> str:
    """Render one configuration through jinja, exactly as aiter's compile() does.

    The keyword names below are aiter's (pa_ragged.py:47-61); ``fp8_kv_dtype``
    takes the config's ``kv_cache_dtype`` because that is the spelling the name
    hash uses. ``version`` is pinned to GOLDEN rather than read from QKV_VERSION:
    an env var that silently swaps in an experimental kernel has no place in a
    build whose output ships in a wheel.
    """
    ctx = dict(
        func_name=name,
        gqa_ratio=cfg["gqa_ratio"],
        head_size=cfg["head_size"],
        npar_loops=cfg["npar_loops"],
        dtype=cfg["dtype"],
        kv_dtype=cfg["kv_dtype"],
        fp8_kv_dtype=cfg["kv_cache_dtype"],
        out_dtype=cfg["out_dtype"],
        block_size=cfg["block_size"],
        partition_size=PARTITION_SIZE,
        mtp=1,
        # The config carries "true"/"false" strings because the name hash is
        # built from their spelling. Passed raw, jinja treats any non-empty
        # string as truthy, so a bare "false" would compile the alibi path in
        # and then read a null slopes pointer on the GPU.
        alibi_enabled=(str(cfg["alibi_enabled"]).lower() == "true"),
        logits_soft_cap_enabled=False,
        version="GOLDEN",
    )

    try:
        from jinja2 import Template
    except ImportError:
        return _render_minimal(template_text, ctx)
    return Template(template_text).render(**ctx)


# The template uses a closed, tiny slice of jinja: `{{name}}`, inline
# `{{"a" if cond else "b"}}` chains, and one `{% if x == 'lit' %}/{% else %}/
# {% endif %}` block. Reimplementing that here keeps jinja2 off the CI image's
# critical path -- an install step is one more thing to fail in a container we
# do not control. The rule is that this must never guess: anything outside the
# grammar below aborts the build and tells you to install jinja2, so a template
# that grows a new construct fails loudly instead of emitting wrong kernels.
_IF_BLOCK = re.compile(
    r"\{%\s*if\s+(\w+)\s*==\s*'([^']*)'\s*%\}(.*?)"
    r"\{%\s*else\s*%\}(.*?)\{%\s*endif\s*%\}",
    re.DOTALL,
)
_COND = re.compile(
    r"^\"([^\"]*)\"\s+if\s+(\w+)\s*(==\s*'([^']*)'|>\s*(\d+))?\s+else\s+(.*)$"
)


def _unsupported(construct: str):
    raise SystemExit(
        "gen_pa_ragged.py: aiter's pa_ragged.cpp.jinja uses a construct the "
        "built-in renderer does not implement:\n\n    " + construct.strip() +
        "\n\nThis renderer is deliberately strict rather than approximate. "
        "Install jinja2 (python3 -m pip install jinja2) to render the template "
        "with aiter's own engine, then teach _render_minimal the construct."
    )


def _eval_expr(expr: str, ctx: dict) -> str:
    """Evaluate one `{{...}}` body: a bare name or an `"a" if cond else ...` chain."""
    expr = expr.strip()
    if expr in ctx:
        return str(ctx[expr])
    m = _COND.match(expr)
    if not m:
        _unsupported("{{" + expr + "}}")
    then, var, _, literal, number, rest = m.groups()
    if var not in ctx:
        _unsupported("{{" + expr + "}}  (unknown variable '" + var + "')")
    value = ctx[var]
    if literal is not None:
        taken = str(value) == literal
    elif number is not None:
        taken = int(value) > int(number)
    else:
        taken = bool(value)
    if taken:
        return then
    rest = rest.strip()
    if rest.startswith('"') and rest.endswith('"'):
        return rest[1:-1]
    return _eval_expr(rest, ctx)


def _render_minimal(text: str, ctx: dict) -> str:
    text = _IF_BLOCK.sub(
        lambda m: m.group(3) if str(ctx.get(m.group(1))) == m.group(2) else m.group(4),
        text,
    )
    text = re.sub(r"\{\{(.*?)\}\}", lambda m: _eval_expr(m.group(1), ctx), text)
    leftover = re.search(r"\{%.*?%\}|\{\{.*?\}\}", text, re.DOTALL)
    if leftover:
        _unsupported(leftover.group(0))
    return text


def verify_rendered_alibi(source: str, cfg: dict, name: str) -> None:
    """Fail the build if the emitted kernel disagrees with the requested config.

    Carried over from the prebuild script. A mis-rendered alibi flag is
    invisible until the GPU faults, and it is cheap to read the template
    argument back rather than trust the round trip.
    """
    want = str(cfg["alibi_enabled"]).lower()
    marker = "paged_attention_ll4mi_QKV_mfma16_kernel<"
    start = source.find(marker)
    if start == -1:
        return
    args = [a.strip() for a in source[start + len(marker):source.index(">", start)].split(",")]
    literals = [a for a in args if a in ("true", "false")]
    if literals and literals[0] != want:
        raise SystemExit(
            f"{name}: kernel rendered with ALIBI_ENABLED={literals[0]} but the "
            f"configuration asked for {want}. The alibi path dereferences a "
            f"slopes pointer the caller passes as null, so this would fault."
        )


def emit_dispatch_header(names: list[str]) -> str:
    decls = "\n".join(f"void {n}({_DECL_PARAMS});" for n in names)
    entries = " \\\n".join(f'  X("{n}", {n})' for n in names)
    return f"""// Generated by scripts/gen_pa_ragged.py -- do not edit.
//
// Declares the ahead-of-time compiled paged-attention entry points and lists
// them as an X-macro, which csrc/ffi/kv/paged_attention_ja.cu expands into a
// static name -> function table. The names are the md5 configuration keys
// jax_aiter.kv.pa_config.func_name computes, so the Python caller, the FFI
// attribute and these symbols all agree by construction.
#pragma once

extern "C" {{
{decls}
}}

#define JA_PA_KERNEL_LIST(X) \\
{entries}

#define JA_PA_KERNEL_COUNT {len(names)}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--aiter-dir",
        default=str(REPO / "third_party" / "aiter"),
        help="aiter checkout to render the template from",
    )
    parser.add_argument("--out-dir", help="directory to write sources into")
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--npar-loops",
        type=int,
        default=1,
        help="ceil(max_num_partitions / 64); 1 covers max_seq_len up to 16384",
    )
    parser.add_argument(
        "--print-signature",
        action="store_true",
        help="print the template's extern-block hash and exit (for updating "
        "EXPECTED_SIGNATURE_SHA256 after an aiter bump)",
    )
    args = parser.parse_args()

    tpl = template_path(Path(args.aiter_dir))
    if not tpl.is_file():
        raise SystemExit(f"template not found: {tpl} (is third_party/aiter checked out?)")
    text = tpl.read_text()

    if args.print_signature:
        print(signature_sha256(text))
        return 0

    check_signature(text)

    if not args.out_dir:
        parser.error("--out-dir is required unless --print-signature is given")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = default_configs(args.head_size, args.block_size, args.npar_loops)
    names: list[str] = []
    for cfg in configs:
        name = config_func_name(cfg)
        source = render(text, cfg, name)
        verify_rendered_alibi(source, cfg, name)
        (out_dir / f"{name}.cpp").write_text(source)
        names.append(name)
        print(f"  {name}  {cfg['dtype']} gqa={cfg['gqa_ratio']}")

    (out_dir / "pa_dispatch_generated.h").write_text(emit_dispatch_header(names))
    print(f"rendered {len(names)} configuration(s) into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
