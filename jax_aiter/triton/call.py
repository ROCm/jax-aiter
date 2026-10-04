# SPDX-License-Identifier: Apache-2.0
"""Lower a compiled Triton kernel to TritonDispatchJA."""

from __future__ import annotations

import functools
import struct
import threading
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Primitive
from jax.interpreters import mlir
from jax.interpreters import xla as _xla

from .bridge import ensure_target_registered, max_devices, register_kernel
from .compile import abi_kind, compile_hsaco, compile_key, prepare_aiter_triton

_INPUT_BUFFER = 0
_OUTPUT_BUFFER = 1
_SCALAR = 2

_COMPILE_LOCK = threading.Lock()
_COMPILE_CACHE: dict[str, tuple[int, int]] = {}
_KERNELS: dict[str, Any] = {}


@dataclass(frozen=True)
class TritonLaunch:
    kernel_id: str
    runtime_types: tuple[tuple[str, str], ...]
    constexprs: tuple[tuple[str, Any], ...]
    options: tuple[tuple[str, Any], ...]
    grid: tuple[int, int, int]
    kinds: tuple[str, ...]
    output_positions: tuple[int, ...]


def _kernel_id(fn) -> str:
    return f"{fn.__module__}.{getattr(fn, '__name__', type(fn).__name__)}"


def _scalar_to_int64(triton_type: str, value: Any) -> int:
    if triton_type in ("fp32", "f32"):
        packed = struct.pack("<f", float(value))
        return int(struct.unpack("<I", packed)[0])
    if triton_type in ("fp64", "f64"):
        packed = struct.pack("<d", float(value))
        return int(struct.unpack("<Q", packed)[0])
    return int(value)


def _ensure_compiled(fn, launch: TritonLaunch) -> tuple[int, int]:
    cache_key = repr(
        (
            launch.kernel_id,
            launch.runtime_types,
            launch.constexprs,
            launch.options,
        )
    )
    with _COMPILE_LOCK:
        cached = _COMPILE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        kernel = compile_hsaco(
            fn,
            runtime_types=dict(launch.runtime_types),
            constexprs=dict(launch.constexprs),
            options=dict(launch.options),
        )
        hsaco = kernel.asm.get("hsaco") or kernel.kernel
        key = compile_key(bytes(hsaco), kernel.name)
        abi = [abi_kind(typ) for _name, typ in launch.runtime_types]
        devices = min(len(jax.local_devices()), max_devices())
        index = register_kernel(
            bytes(hsaco),
            kernel.name,
            devices,
            int(kernel.metadata.num_warps),
            int(kernel.metadata.num_ctas),
            int(kernel.metadata.shared),
            int(kernel.metadata.warp_size),
            key,
            abi,
            name=kernel.name,
        )
        _COMPILE_CACHE[cache_key] = (index, key)
        return index, key


triton_call_p = Primitive("jax_aiter_triton_call")
triton_call_p.multiple_results = True


@triton_call_p.def_abstract_eval
def _abstract_eval(*_args, out_avals, **_params):
    return list(out_avals)


triton_call_p.def_impl(functools.partial(_xla.apply_primitive, triton_call_p))


def _lowering(
    ctx,
    *array_args,
    launch: TritonLaunch,
    out_avals,
    input_output_aliases,
    scalar_values,
):
    fn = _KERNELS[launch.kernel_id]
    compile_index, compile_key_value = _ensure_compiled(fn, launch)

    arg_kinds: list[int] = []
    arg_indices: list[int] = []
    packed: list[int] = []
    input_index = 0
    output_index = 0
    output_set = set(launch.output_positions)
    aliased_inputs = {src for src, _ in input_output_aliases}
    logical_inputs = [
        index for index in range(len(array_args)) if index not in aliased_inputs
    ]
    scalar_iter = iter(scalar_values)

    for position, (_name, triton_type) in enumerate(launch.runtime_types):
        kind = launch.kinds[position]
        if kind == "scalar":
            arg_kinds.append(_SCALAR)
            arg_indices.append(0)
            packed.append(_scalar_to_int64(triton_type, next(scalar_iter)))
            continue
        if position in output_set:
            arg_kinds.append(_OUTPUT_BUFFER)
            arg_indices.append(output_index)
            packed.append(0)
            output_index += 1
        else:
            arg_kinds.append(_INPUT_BUFFER)
            arg_indices.append(logical_inputs[input_index])
            packed.append(0)
            input_index += 1

    if output_index != len(out_avals):
        raise ValueError("Triton output count mismatch")

    target_name = ensure_target_registered()
    lowering = jax.ffi.ffi_lowering(
        target_name,
        operand_output_aliases=dict(input_output_aliases),
    )
    return lowering(
        ctx,
        *array_args,
        compile_index=np.int64(compile_index),
        compile_key=np.int64(compile_key_value),
        grid_x=np.int64(launch.grid[0]),
        grid_y=np.int64(launch.grid[1]),
        grid_z=np.int64(launch.grid[2]),
        arg_kinds=np.asarray(arg_kinds, dtype=np.int32),
        arg_indices=np.asarray(arg_indices, dtype=np.int32),
        scalar_values=np.asarray(packed, dtype=np.int64),
    )


mlir.register_lowering(triton_call_p, _lowering, platform="rocm")


def triton_call(
    *array_args: jax.Array,
    fn,
    runtime_types: dict[str, str],
    constexprs: dict[str, Any],
    kinds: list[str] | tuple[str, ...],
    scalars: dict[str, Any],
    out_shape,
    output_names: tuple[str, ...] | list[str],
    grid: tuple[int, int, int],
    options: dict[str, Any] | None = None,
    input_output_aliases: dict[int, int] | None = None,
):
    """Launch one compiled Triton kernel through TritonDispatchJA."""
    prepare_aiter_triton()
    ensure_target_registered()
    kernel_id = _kernel_id(fn)
    _KERNELS[kernel_id] = fn

    runtime_items = tuple(runtime_types.items())
    kind_tuple = tuple(kinds)
    if len(kind_tuple) != len(runtime_items):
        raise ValueError("kinds must match runtime_types")

    flat_outputs, output_tree = jax.tree_util.tree_flatten(out_shape)
    output_names = tuple(output_names)
    if len(output_names) != len(flat_outputs):
        raise ValueError("output_names must match out_shape")
    name_to_pos = {name: index for index, (name, _) in enumerate(runtime_items)}
    output_positions = tuple(name_to_pos[name] for name in output_names)

    scalar_names = [
        name
        for (name, _typ), kind in zip(runtime_items, kind_tuple)
        if kind == "scalar"
    ]
    missing = set(scalar_names) - set(scalars)
    extra = set(scalars) - set(scalar_names)
    if missing or extra:
        raise ValueError(
            f"Triton scalar mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    scalar_values = tuple(scalars[name] for name in scalar_names)

    launch = TritonLaunch(
        kernel_id=kernel_id,
        runtime_types=runtime_items,
        constexprs=tuple(sorted(constexprs.items())),
        options=tuple(sorted((options or {}).items())),
        grid=tuple(int(x) for x in grid),
        kinds=kind_tuple,
        output_positions=output_positions,
    )
    out_avals = tuple(
        jax.core.ShapedArray(tuple(spec.shape), spec.dtype) for spec in flat_outputs
    )
    results = triton_call_p.bind(
        *tuple(jnp.asarray(arg) for arg in array_args),
        launch=launch,
        out_avals=out_avals,
        input_output_aliases=tuple(sorted((input_output_aliases or {}).items())),
        scalar_values=scalar_values,
    )
    return jax.tree_util.tree_unflatten(output_tree, results)
