# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Assertions for the in-place KV mutation contract through XLA FFI (M0).

The contract these enforce: a KV pool bound as both operand and result of an FFI
call must be mutated in place, with no pool-sized replacement allocation and no
pool-sized copy, for the lifetime of a decode run.

Address identity is deliberately *not* the primary invariant. ``unsafe_buffer_-
pointer()`` is unstable API and a poor portable guarantee, and under the
collective-memory path a replacement buffer would still be correctly placed. The
primary evidence is therefore the alias declaration surviving into compiled HLO,
the absence of a pool-sized temp allocation, and flat live memory across a long
run. ``pool_shard_pointers`` remains available as a diagnostic, and becomes
mandatory only under the external-registration fallback.

The textual markers below were read off real ROCm output (jaxlib 0.10.2,
ROCm 7.2.3, gfx942); each check falls back to a structural signal where one
exists so a compiler-side spelling change degrades to a clear failure rather
than a silent pass.
"""

from __future__ import annotations

import contextlib
import dataclasses
import re
import warnings
from typing import Any, Callable, Sequence

import jax
import numpy as np

# StableHLO, before optimisation: the aliasing intent as JAX lowered it.
_LOWERED_ALIAS_RE = re.compile(
    r"stablehlo\.output_operand_alias<[^>]*operand_index\s*=\s*(\d+)"
)
# Donation at the jit boundary, as an argument attribute.
_LOWERED_DONATION_RE = re.compile(r"tf\.aliasing_output\s*=\s*(\d+)")
# Optimised HLO: the FFI call still declares the alias. Both spellings occur --
# `{}: (n, {})` for a single result and `{i}: (n, {})` per element when the call
# returns a tuple -- so the output index is captured as a possibly-empty string.
_COMPILED_ALIAS_TAIL_RE = re.compile(r"output_to_operand_aliasing=(.*)$", re.M)
_COMPILED_ENTRY_TAIL_RE = re.compile(r"input_output_alias=(\{.*?\})\s*,\s*\w+=", re.S)
_ALIAS_PAIR_RE = re.compile(r"\{(\d*)\}\s*:\s*\((\d+)\s*,")


class AliasContractError(AssertionError):
    """Raised when the in-place KV mutation contract is not upheld."""


@dataclasses.dataclass
class AliasEvidence:
    """Everything observed about one compiled step, for reporting.

    Two index spaces appear here and conflating them is easy, because for a
    single-pool op they coincide:

      * ``donated_argnum`` indexes the *jit* arguments, which is what
        ``donate_argnums`` refers to.
      * ``ffi_operand_index`` indexes the *custom call's* operands, which is what
        ``input_output_aliases`` refers to.

    For ``append_kv`` the K pool is jit argument 0 but FFI operand 5, so they must
    be tracked separately.

    ``ok`` deliberately excludes pointer stability, which is a diagnostic.
    """

    target: str
    pool_bytes: int
    donated_argnum: int = 0
    ffi_operand_index: int = 0
    lowered_alias_operands: tuple[int, ...] = ()
    lowered_donated_args: tuple[int, ...] = ()
    compiled_alias_pairs: tuple[tuple[str, int], ...] = ()
    compiled_entry_pairs: tuple[tuple[str, int], ...] = ()
    alias_size_in_bytes: int | None = None
    temp_size_in_bytes: int | None = None
    pool_sized_copies: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()

    @property
    def compiled_alias_operands(self) -> tuple[int, ...]:
        return tuple(operand for _, operand in self.compiled_alias_pairs)

    @property
    def compiled_entry_args(self) -> tuple[int, ...]:
        return tuple(operand for _, operand in self.compiled_entry_pairs)

    @property
    def ok(self) -> bool:
        return not self.failures

    def report(self) -> str:
        lines = [f"alias evidence for {self.target!r} (pool {self.pool_bytes} bytes):"]
        lines.append(f"  jit arg / ffi operand:       {self.donated_argnum} / {self.ffi_operand_index}")
        lines.append(f"  lowered alias on operands:   {self.lowered_alias_operands}")
        lines.append(f"  lowered donated args:        {self.lowered_donated_args}")
        lines.append(f"  compiled call alias pairs:   {self.compiled_alias_pairs}")
        lines.append(f"  compiled entry alias pairs:  {self.compiled_entry_pairs}")
        lines.append(f"  alias_size_in_bytes:         {self.alias_size_in_bytes}")
        lines.append(f"  temp_size_in_bytes:          {self.temp_size_in_bytes}")
        lines.append(f"  pool-sized copies:           {self.pool_sized_copies or 'none'}")
        if self.failures:
            lines.append("  FAILURES:")
            lines.extend(f"    - {f}" for f in self.failures)
        else:
            lines.append("  all required checks passed")
        return "\n".join(lines)


def _hlo_shape_prefix(aval_like) -> str:
    """Render an array's HLO shape prefix, e.g. ``f32[64,128]``."""
    dtype_map = {
        "float32": "f32",
        "float16": "f16",
        "bfloat16": "bf16",
        "float64": "f64",
        "int32": "s32",
        "int64": "s64",
        "int8": "s8",
        "uint16": "u16",
        "uint32": "u32",
    }
    name = np.dtype(aval_like.dtype).name
    elem = dtype_map.get(name, name)
    dims = ",".join(str(d) for d in aval_like.shape)
    return f"{elem}[{dims}]"


def _find_pool_sized_copies(compiled_text: str, pool_shape_prefix: str) -> tuple[str, ...]:
    """Return copy-family instructions whose result has the pool's shape.

    A pool-sized copy is the failure this is really hunting: it means XLA
    honoured the alias only by first duplicating the buffer, which costs the
    bandwidth the whole design is trying to avoid.
    """
    found = []
    for line in compiled_text.splitlines():
        stripped = line.strip()
        if pool_shape_prefix not in stripped:
            continue
        # Match "= <shape>{layout} copy(" and the async copy pair.
        if re.search(
            r"=\s*" + re.escape(pool_shape_prefix) + r"\{[^}]*\}\s*"
            r"(copy|copy-start|copy-done|dynamic-update-slice)\(",
            stripped,
        ):
            found.append(stripped.split(" metadata=")[0])
    return tuple(found)


def _alias_pairs(text: str, tail_re: re.Pattern) -> tuple[tuple[str, int], ...]:
    """Extract ``(output_index, operand_index)`` pairs from an alias attribute."""
    pairs: list[tuple[str, int]] = []
    for tail in tail_re.findall(text):
        for out_idx, operand in _ALIAS_PAIR_RE.findall(tail):
            pairs.append((out_idx, int(operand)))
    return tuple(pairs)


def collect_alias_evidence(
    jitted: Any,
    *args,
    pool_argnum: int = 0,
    ffi_operand_index: int | None = None,
    target: str,
    expect_entry_donation: bool = True,
) -> AliasEvidence:
    """Compile ``jitted`` and gather every alias signal from the HLO.

    Args:
        jitted: a ``jax.jit``-wrapped step, normally with ``donate_argnums``
            covering ``pool_argnum``.
        *args: concrete arguments to lower with.
        pool_argnum: which *jit argument* is the KV pool. Used for the donation
            checks and to size the pool.
        ffi_operand_index: which *custom call operand* the pool is, if it differs
            from ``pool_argnum``. For a single-pool op the two coincide and this
            can be left unset; ``append_kv`` passes 5 and 6 for K and V while
            they are jit arguments 0 and 1.
        target: the FFI target name expected to carry the alias.
        expect_entry_donation: require the pool to be donated at the jit
            boundary as well as aliased inside the call.

    Returns:
        An :class:`AliasEvidence` whose ``failures`` is empty on success.
    """
    operand_index = pool_argnum if ffi_operand_index is None else ffi_operand_index

    pool = args[pool_argnum]
    pool_bytes = int(np.prod(pool.shape)) * np.dtype(pool.dtype).itemsize
    shape_prefix = _hlo_shape_prefix(pool)

    lowered = jitted.lower(*args)
    lowered_text = lowered.as_text()
    compiled = lowered.compile()
    compiled_text = compiled.as_text()

    ev = AliasEvidence(
        target=target,
        pool_bytes=pool_bytes,
        donated_argnum=pool_argnum,
        ffi_operand_index=operand_index,
    )
    ev.lowered_alias_operands = tuple(
        int(m) for m in _LOWERED_ALIAS_RE.findall(lowered_text)
    )
    ev.lowered_donated_args = tuple(
        int(m) for m in _LOWERED_DONATION_RE.findall(lowered_text)
    )
    ev.compiled_alias_pairs = _alias_pairs(compiled_text, _COMPILED_ALIAS_TAIL_RE)
    ev.compiled_entry_pairs = _alias_pairs(compiled_text, _COMPILED_ENTRY_TAIL_RE)
    ev.pool_sized_copies = _find_pool_sized_copies(compiled_text, shape_prefix)

    try:
        ma = compiled.memory_analysis()
        ev.alias_size_in_bytes = getattr(ma, "alias_size_in_bytes", None)
        ev.temp_size_in_bytes = getattr(ma, "temp_size_in_bytes", None)
    except Exception:  # memory_analysis is best-effort across backends
        pass

    failures = []

    if target not in compiled_text:
        failures.append(
            f"{target} absent from compiled HLO -- the call was eliminated or renamed"
        )
    if operand_index not in ev.lowered_alias_operands:
        failures.append(
            f"lowered HLO declares no alias on FFI operand {operand_index}; "
            f"saw {ev.lowered_alias_operands}. Is input_output_aliases set?"
        )
    if operand_index not in ev.compiled_alias_operands:
        failures.append(
            f"compiled HLO declares no alias on FFI operand {operand_index}; "
            f"saw {ev.compiled_alias_operands}. The alias did not survive optimisation."
        )
    if expect_entry_donation:
        if pool_argnum not in ev.lowered_donated_args:
            failures.append(
                f"jit argument {pool_argnum} is not donated; "
                f"add donate_argnums={pool_argnum}"
            )
        if pool_argnum not in ev.compiled_entry_args:
            failures.append(
                f"compiled HLO module does not record jit argument {pool_argnum} "
                f"as donated; saw {ev.compiled_entry_args}"
            )
    if ev.pool_sized_copies:
        failures.append(
            f"{len(ev.pool_sized_copies)} pool-sized copy instruction(s) in "
            f"compiled HLO: {ev.pool_sized_copies}"
        )
    if ev.temp_size_in_bytes is not None and ev.temp_size_in_bytes >= pool_bytes:
        failures.append(
            f"temp_size_in_bytes {ev.temp_size_in_bytes} >= pool size {pool_bytes}, "
            f"which is consistent with a pool-sized replacement allocation"
        )
    if ev.alias_size_in_bytes is not None and ev.alias_size_in_bytes < pool_bytes:
        failures.append(
            f"alias_size_in_bytes {ev.alias_size_in_bytes} < pool size {pool_bytes}, "
            f"so the pool is not fully aliased"
        )

    ev.failures = tuple(failures)
    return ev


def assert_in_place_mutation(
    jitted: Any,
    *args,
    pool_argnum: int = 0,
    ffi_operand_index: int | None = None,
    target: str,
    expect_entry_donation: bool = True,
) -> AliasEvidence:
    """Raise :class:`AliasContractError` unless the pool is mutated in place."""
    ev = collect_alias_evidence(
        jitted,
        *args,
        pool_argnum=pool_argnum,
        ffi_operand_index=ffi_operand_index,
        target=target,
        expect_entry_donation=expect_entry_donation,
    )
    if not ev.ok:
        raise AliasContractError(ev.report())
    return ev


@contextlib.contextmanager
def donation_warnings_as_errors():
    """Turn JAX's "donated buffers were not usable" warning into an exception.

    A silently declined donation is the dangerous case: the step still computes
    the right answer while quietly allocating a replacement buffer every
    iteration, so it presents as a throughput regression rather than a failure.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r".*[Dd]onat.*")
        warnings.filterwarnings("error", message=r".*buffer donation.*")
        yield


def live_bytes(device=None) -> int:
    """Bytes currently in use on ``device`` (default: first local device)."""
    dev = device or jax.local_devices()[0]
    stats = dev.memory_stats() or {}
    return int(stats.get("bytes_in_use", 0))


def assert_flat_live_memory(
    step: Callable[..., Any],
    init_pool: Any,
    other_args: Sequence[Any],
    *,
    steps: int = 512,
    warmup: int = 8,
    tolerance_bytes: int | None = None,
    device=None,
) -> tuple[int, int, Any]:
    """Run ``step`` in a donation loop and require live memory not to grow.

    ``step`` is called as ``step(pool, *other_args)`` and must return the new
    pool. Growth proportional to the step count is the signature of a
    replacement allocation per iteration.

    Returns:
        ``(baseline_bytes, final_bytes, final_pool)``.
    """
    dev = device or jax.local_devices()[0]
    pool_bytes = int(np.prod(init_pool.shape)) * np.dtype(init_pool.dtype).itemsize
    # Half a pool is a generous ceiling: one replacement allocation would exceed it.
    tol = tolerance_bytes if tolerance_bytes is not None else max(pool_bytes // 2, 1 << 20)

    pool = init_pool
    for _ in range(warmup):
        pool = step(pool, *other_args)
    pool.block_until_ready()
    baseline = live_bytes(dev)

    for _ in range(steps):
        pool = step(pool, *other_args)
    pool.block_until_ready()
    final = live_bytes(dev)

    if final - baseline > tol:
        raise AliasContractError(
            f"live memory grew {final - baseline} bytes over {steps} steps "
            f"(baseline {baseline}, final {final}, tolerance {tol}, "
            f"pool {pool_bytes}). A per-step replacement allocation is the "
            f"likely cause."
        )
    return baseline, final, pool


def pool_shard_pointers(arr) -> tuple[int, ...]:
    """Per-shard device pointers, as a diagnostic only.

    ``unsafe_buffer_pointer()`` is valid only for single-device arrays, so this
    iterates ``addressable_shards``. Under the external-registration fallback
    this becomes a required invariant; under collective-memory placement it is
    just a cheap smoke check.
    """
    ptrs = []
    for shard in arr.addressable_shards:
        try:
            ptrs.append(int(shard.data.unsafe_buffer_pointer()))
        except Exception:
            ptrs.append(0)
    return tuple(ptrs)
