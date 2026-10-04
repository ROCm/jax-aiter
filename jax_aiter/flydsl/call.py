# SPDX-License-Identifier: Apache-2.0
"""Lower FlyDSL ``@jit`` launchers to a typed XLA FFI custom call.

This is intentionally a small, first-party bridge. FlyDSL compilation happens
during JAX lowering; execution calls a process-local compiled function through
``FlysdlDispatchJA`` with no Python on the dispatch path.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import struct
import sysconfig
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import core
from jax.extend.core import Primitive
from jax.interpreters import mlir
from jax.interpreters import xla as _xla

from .bridge import (
    ensure_target_registered,
    hip_get_device,
    hip_set_device,
    register_compile,
)

_INPUT_BUFFER = 0
_OUTPUT_BUFFER = 1
_SCALAR = 2

_SCALAR_BITCAST = {
    "Int32": None,
    "Float32": ("<f", "<I"),
}

_COMPILE_CACHE: dict[str, tuple[str, int, int]] = {}
_ARTIFACTS: list[Any] = []
_COMPILE_LOCK = threading.Lock()
_STATIC_MEMREF_CLASS = None
_STATIC_MEMREF_LOCK = threading.Lock()
_TOOLKIT_PATCHED = False
_TOOLKIT_LOCK = threading.Lock()


@dataclass(frozen=True)
class _LauncherLayout:
    names: tuple[str, ...]
    kinds: tuple[str, ...]
    scalar_types: tuple[str, ...]

    @property
    def num_buffers(self) -> int:
        return sum(kind == "buffer" for kind in self.kinds)

    @property
    def scalar_type_by_name(self) -> dict[str, str]:
        return {
            name: scalar_type
            for name, kind, scalar_type in zip(
                self.names, self.kinds, self.scalar_types
            )
            if kind == "scalar"
        }


def _ensure_gpu_initialized() -> None:
    # FlyDSL imports torch from its compiler argument module. Initialize JAX's
    # ROCm backend first so a GPU-enabled torch build cannot win HIP startup.
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("FlyDSL MXFP4 requires a ROCm GPU")

    import flydsl
    from flydsl.runtime.device import get_rocm_arch

    if flydsl.__version__ != "0.3.0":
        raise RuntimeError(
            f"jax-aiter MXFP4 requires FlyDSL 0.3.0, got {flydsl.__version__}"
        )
    arch = str(get_rocm_arch())
    if arch != "gfx950":
        raise RuntimeError(f"jax-aiter MXFP4 requires gfx950, got {arch}")


def _valid_rocm_toolkit(path: Path) -> bool:
    return (
        (path / "llvm" / "bin" / "ld.lld").is_file()
        and (path / "amdgcn" / "bitcode").is_dir()
    )


def _rocm_toolkit_path() -> Path:
    """Return an MLIR-compatible ROCm toolkit, including wheel installs."""
    for raw_path in (
        os.environ.get("FLYDSL_ROCM_TOOLKIT_PATH"),
        os.environ.get("ROCM_PATH"),
        "/opt/rocm",
    ):
        if raw_path:
            candidate = Path(raw_path).expanduser().resolve()
            if _valid_rocm_toolkit(candidate):
                return candidate

    purelib = Path(sysconfig.get_paths()["purelib"])
    llvm_root = purelib / "_rocm_sdk_devel" / "lib" / "llvm"
    real_lld = llvm_root / "bin" / "ld.lld"
    real_amdgcn = llvm_root / "amdgcn"
    if not real_lld.is_file() or not (real_amdgcn / "bitcode").is_dir():
        raise RuntimeError(
            "Unable to locate an MLIR-compatible ROCm toolkit. Set "
            "FLYDSL_ROCM_TOOLKIT_PATH to a directory containing "
            "llvm/bin/ld.lld and amdgcn/bitcode."
        )

    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    )
    toolkit = cache_root / "jax-aiter" / "rocm-toolkit"
    lld_wrapper = toolkit / "llvm" / "bin" / "ld.lld"
    amdgcn_link = toolkit / "amdgcn"
    lld_wrapper.parent.mkdir(parents=True, exist_ok=True)
    if not amdgcn_link.exists():
        amdgcn_link.symlink_to(real_amdgcn, target_is_directory=True)
    if not lld_wrapper.exists():
        lld_wrapper.write_text(
            "#!/bin/sh\n"
            f'exec "{real_lld}" "$@"\n',
            encoding="utf-8",
        )
        lld_wrapper.chmod(0o755)
    if not _valid_rocm_toolkit(toolkit):
        raise RuntimeError(f"Failed to construct ROCm toolkit shim at {toolkit}")
    return toolkit


def _patch_flydsl_rocm_toolkit() -> None:
    """Teach FlyDSL 0.3.x about the rocm-sdk wheel directory layout."""
    global _TOOLKIT_PATCHED
    if _TOOLKIT_PATCHED:
        return
    with _TOOLKIT_LOCK:
        if _TOOLKIT_PATCHED:
            return
        from flydsl.compiler.backends.rocm import RocmBackend

        toolkit = _rocm_toolkit_path()
        original = RocmBackend._pipeline_parts

        def pipeline_parts(self, *, compile_hints):
            fragments, binary = original(self, compile_hints=compile_hints)
            marker = "gpu-module-to-binary{"
            if marker not in binary:
                raise RuntimeError(
                    f"Unexpected FlyDSL ROCm binary pipeline: {binary}"
                )
            binary = binary.replace(
                marker, f"{marker}toolkit={toolkit} ", 1
            )
            return fragments, binary

        RocmBackend._pipeline_parts = pipeline_parts
        _TOOLKIT_PATCHED = True


def _inspect_launcher(launcher) -> _LauncherLayout:
    from flydsl.compiler.jit_function import JitFunction
    import flydsl.expr as fx

    if not isinstance(launcher, JitFunction):
        raise TypeError(
            "flydsl_call expects a FlyDSL @jit launcher, got "
            f"{type(launcher).__name__}"
        )

    signature = inspect.signature(launcher.func)
    names: list[str] = []
    kinds: list[str] = []
    scalar_types: list[str] = []

    for name, parameter in signature.parameters.items():
        annotation = parameter.annotation
        annotation_name = getattr(annotation, "__name__", "")
        if isinstance(annotation, str):
            annotation_name = annotation.split(".")[-1].strip("\"' ")

        if annotation is fx.Stream or annotation_name == "Stream":
            kind, scalar_type = "stream", ""
        elif annotation is fx.Tensor or annotation_name == "Tensor":
            kind, scalar_type = "buffer", ""
        elif annotation_name in _SCALAR_BITCAST:
            kind, scalar_type = "scalar", annotation_name
        elif annotation is inspect.Parameter.empty:
            raise TypeError(
                f"FlyDSL launcher parameter {name!r} must have an explicit "
                "fx.Tensor, fx.Int32, fx.Float32, or fx.Stream annotation"
            )
        else:
            raise TypeError(
                f"Cannot classify FlyDSL launcher parameter {name!r} with "
                f"annotation {annotation!r}"
            )
        names.append(name)
        kinds.append(kind)
        scalar_types.append(scalar_type)

    stream_positions = [i for i, kind in enumerate(kinds) if kind == "stream"]
    if stream_positions != [len(kinds) - 1]:
        raise ValueError(
            "FlyDSL launcher must declare exactly one final fx.Stream parameter"
        )

    return _LauncherLayout(tuple(names), tuple(kinds), tuple(scalar_types))


def _scalar_to_int64(type_name: str, value: Any) -> int:
    formats = _SCALAR_BITCAST[type_name]
    if formats is None:
        value = int(value)
        if not -(1 << 31) <= value < (1 << 31):
            raise OverflowError(f"FlyDSL Int32 scalar is out of range: {value}")
        return value
    pack_format, unpack_format = formats
    return int(struct.unpack(unpack_format, struct.pack(pack_format, value))[0])


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [0] * len(shape)
    stride = 1
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = stride
        stride *= int(shape[index])
    return tuple(strides)


def _static_memref_class():
    global _STATIC_MEMREF_CLASS
    if _STATIC_MEMREF_CLASS is not None:
        return _STATIC_MEMREF_CLASS

    with _STATIC_MEMREF_LOCK:
        if _STATIC_MEMREF_CLASS is not None:
            return _STATIC_MEMREF_CLASS

        from flydsl.compiler.jit_argument import (
            JitArgumentRegistry,
            MemRefJitArg,
        )
        from flydsl.expr.typing import Tensor

        class StaticMemRefJitArg(MemRefJitArg):
            """Shape-only contiguous memref used solely while emitting MLIR."""

            def __init__(self, shape, dtype):
                dtype = np.dtype(dtype)
                self.numpy_dtype = dtype
                super().__init__(
                    element_bits=dtype.itemsize * 8,
                    shape=tuple(int(dimension) for dimension in shape),
                    strides=_row_major_strides(tuple(shape)),
                    dtype=dtype.str,
                    dynamic_layout=False,
                )

            @property
            def element_type(self):
                from flydsl._mlir import ir
                from flydsl._mlir.extras import types as types

                dtype = self.numpy_dtype
                if dtype == np.dtype(jnp.bfloat16):
                    return types.bf16()
                if dtype == np.dtype(np.float16):
                    return types.f16()
                if dtype == np.dtype(np.float32):
                    return types.f32()
                if dtype == np.dtype(np.float64):
                    return types.f64()
                if dtype in (np.dtype(np.uint8), np.dtype(np.int8)):
                    return ir.IntegerType.get_signless(8)
                if dtype in (np.dtype(np.uint16), np.dtype(np.int16)):
                    return ir.IntegerType.get_signless(16)
                if dtype in (np.dtype(np.uint32), np.dtype(np.int32)):
                    return ir.IntegerType.get_signless(32)
                if dtype in (np.dtype(np.uint64), np.dtype(np.int64)):
                    return ir.IntegerType.get_signless(64)
                if dtype == np.dtype(np.bool_):
                    return ir.IntegerType.get_signless(1)
                raise TypeError(f"Unsupported FlyDSL buffer dtype: {dtype}")

            def __c_abi_spec__(self):
                # ``build_module`` never executes this descriptor. Implementing
                # the protocol keeps FlyDSL's JitArgument validation honest.
                return [(ctypes.c_void_p, lambda _arg, slot: setattr(slot, "value", 0))]

        JitArgumentRegistry.register_jit_arg(StaticMemRefJitArg, Tensor)
        _STATIC_MEMREF_CLASS = StaticMemRefJitArg
        return _STATIC_MEMREF_CLASS


def _shape_memref(shape, dtype):
    return _static_memref_class()(shape, dtype)


def _build_module(launcher, values_by_name: dict[str, Any]):
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import func
    from flydsl.compiler.backends import get_backend
    from flydsl.compiler.jit_argument import (
        convert_to_jit_arguments,
        resolve_signature,
    )
    from flydsl.compiler.jit_function import _ensure_stream_arg
    from flydsl.compiler.kernel_function import (
        CompilationContext,
        create_gpu_module,
        effective_fastmath_hint,
        func_def_location,
        get_gpu_module_body,
    )
    from flydsl.compiler.protocol import (
        construct_from_ir_values,
        get_ir_types,
    )
    from flydsl.expr.meta import tracing_context
    from flydsl.expr.utils.arith import fastmath as fastmath_context

    underlying = launcher.func
    signature = resolve_signature(underlying)
    bound = signature.bind(**values_by_name)
    bound.apply_defaults()

    context = ir.Context()
    context.enable_multithreading(False)
    context.load_all_available_dialects()
    context.__enter__()
    try:
        parameter_names, jit_args, dsl_types, constexpr_values = (
            convert_to_jit_arguments(signature, bound)
        )
        has_user_stream = _ensure_stream_arg(jit_args)
        ir_types = get_ir_types(jit_args)
        location = func_def_location(underlying, context)
        module = ir.Module.create(loc=location)
        module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()

        with ir.InsertionPoint(module.body), location:
            backend = get_backend()
            gpu_module = create_gpu_module(
                "kernels", targets=backend.gpu_module_targets()
            )
            function = func.FuncOp(underlying.__name__, (ir_types, []))
            function.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            entry_block = function.add_entry_block()

            with CompilationContext.create() as compile_context:
                compile_context.gpu_module_op = gpu_module
                compile_context.gpu_module_body = get_gpu_module_body(gpu_module)
                with ir.InsertionPoint(entry_block):
                    ir_args = list(entry_block.arguments)
                    if not has_user_stream:
                        compile_context.stream_arg = ir_args[-1]
                    user_jit_args = jit_args[: len(parameter_names)]
                    dsl_args = construct_from_ir_values(
                        dsl_types, user_jit_args, ir_args
                    )
                    named_args = dict(zip(parameter_names, dsl_args))
                    named_args.update(constexpr_values)
                    fastmath = effective_fastmath_hint(
                        CompilationContext.get_compile_hints()
                    )
                    fastmath_scope = (
                        fastmath_context(fastmath)
                        if fastmath is not None
                        else nullcontext()
                    )
                    with tracing_context(underlying), fastmath_scope:
                        underlying(**named_args)
                    func.ReturnOp([])

        if compile_context.link_libs or compile_context.post_load_processors:
            raise NotImplementedError(
                "jax-aiter FlyDSL bridge does not support externally linked kernels"
            )
        result = (module, backend.target.arch)
        context.__exit__(None, None, None)
        return result
    except Exception:
        context.__exit__(*__import__("sys").exc_info())
        raise


def _function_name(module) -> str:
    with module.context:
        for operation in module.body.operations:
            if "llvm.emit_c_interface" in operation.attributes:
                return str(operation.attributes["sym_name"]).strip('"')
    raise RuntimeError("FlyDSL module has no C-interface entry function")


def _compile_module(
    module, arch: str, key: str, report_name: str
) -> tuple[str, int, int]:
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    from flydsl.compiler.jit_executor import CompiledArtifact
    from flydsl.compiler.jit_function import MlirCompiler

    function_name = _function_name(module)
    with module.context:
        source_ir = module.operation.get_asm(enable_debug_info=True)
        compiled = MlirCompiler.compile(
            module, arch=arch, func_name=function_name
        )

    devices = jax.local_devices(backend="gpu")
    if not devices:
        raise RuntimeError("FlyDSL compilation requires a local ROCm device")
    if [device.local_hardware_id for device in devices] != list(range(len(devices))):
        raise RuntimeError(
            "FlyDSL bridge currently requires contiguous local GPU ordinals"
        )

    saved_device = hip_get_device()
    function_ptrs: list[int] = []
    try:
        for device in devices:
            hip_set_device(device.local_hardware_id)
            artifact = CompiledArtifact(compiled, function_name, source_ir)
            function_executable = artifact._get_func_exe()
            pointer = ctypes.cast(function_executable, ctypes.c_void_p).value
            if pointer is None:
                raise RuntimeError("FlyDSL returned a null compiled function")
            function_ptrs.append(pointer)
            _ARTIFACTS.append(artifact)
    finally:
        hip_set_device(saved_device)

    stable_key = int(key[:16], 16) & ((1 << 63) - 1)
    compile_index = register_compile(
        function_ptrs,
        stable_key,
        f"{report_name}:{key[:8]}",
    )
    result = (function_name, compile_index, stable_key)
    _COMPILE_CACHE[key] = result
    return result


flydsl_call_p = Primitive("jax_aiter_flydsl_call")
flydsl_call_p.multiple_results = True


@flydsl_call_p.def_abstract_eval
def _abstract_eval(*_args, out_shapes, **_params):
    return [core.ShapedArray(shape.shape, shape.dtype) for shape in out_shapes]


flydsl_call_p.def_impl(functools.partial(_xla.apply_primitive, flydsl_call_p))


def _lowering(
    ctx,
    *_array_args,
    kernel,
    layout,
    output_positions,
    scalar_pairs,
    input_specs,
    output_specs,
    out_shapes,
    input_output_aliases,
    compile_hints_json,
):
    del out_shapes
    import flydsl.expr as fx
    from flydsl.compiler.kernel_function import CompilationContext

    scalar_values_by_name = dict(scalar_pairs)
    scalar_types = layout.scalar_type_by_name
    output_position_set = frozenset(output_positions)

    values_by_name: dict[str, Any] = {}
    arg_kinds: list[int] = []
    arg_indices: list[int] = []
    packed_scalars: list[int] = []
    input_index = 0
    output_index = 0
    aliased_inputs = {input_index for input_index, _ in input_output_aliases}
    logical_input_indices = [
        index for index in range(len(input_specs)) if index not in aliased_inputs
    ]

    for position, (name, kind) in enumerate(zip(layout.names, layout.kinds)):
        if kind == "stream":
            values_by_name[name] = fx.Stream(None)
            continue
        if kind == "scalar":
            value = scalar_values_by_name[name]
            values_by_name[name] = value
            arg_kinds.append(_SCALAR)
            arg_indices.append(0)
            packed_scalars.append(_scalar_to_int64(scalar_types[name], value))
            continue

        if position in output_position_set:
            shape, dtype = output_specs[output_index]
            values_by_name[name] = _shape_memref(shape, dtype)
            arg_kinds.append(_OUTPUT_BUFFER)
            arg_indices.append(output_index)
            output_index += 1
        else:
            source_index = logical_input_indices[input_index]
            shape, dtype = input_specs[source_index]
            values_by_name[name] = _shape_memref(shape, dtype)
            arg_kinds.append(_INPUT_BUFFER)
            arg_indices.append(source_index)
            input_index += 1
        packed_scalars.append(0)

    if input_index != len(logical_input_indices):
        raise ValueError(
            "FlyDSL launcher did not consume every non-aliased input buffer"
        )
    if output_index != len(output_specs):
        raise ValueError("FlyDSL launcher did not consume every output buffer")

    compile_hints = json.loads(compile_hints_json) if compile_hints_json else None

    def hint_scope():
        if compile_hints is None:
            return nullcontext()
        return CompilationContext.compile_hints(compile_hints)

    with hint_scope():
        module, arch = _build_module(kernel, values_by_name)
    with module.context:
        module_asm = module.operation.get_asm()
    import flydsl

    # Hints steer code generation without appearing in the traced module.
    key_material = f"{flydsl.__version__}:{arch}\0{module_asm}"
    if compile_hints_json:
        key_material += f"\0compile_hints:{compile_hints_json}"
    module_key = hashlib.sha256(key_material.encode()).hexdigest()
    with _COMPILE_LOCK, hint_scope():
        report_name = (
            f"{kernel.func.__module__}.{kernel.func.__qualname__}"
        )
        _function, compile_index, compile_key = _compile_module(
            module, arch, module_key, report_name
        )

    target_name = ensure_target_registered()
    lowering = jax.ffi.ffi_lowering(
        target_name,
        operand_output_aliases=dict(input_output_aliases),
    )
    return lowering(
        ctx,
        *_array_args,
        compile_index=np.int64(compile_index),
        compile_key=np.int64(compile_key),
        arg_kinds=np.asarray(arg_kinds, dtype=np.int32),
        arg_indices=np.asarray(arg_indices, dtype=np.int32),
        scalar_values=np.asarray(packed_scalars, dtype=np.int64),
    )


mlir.register_lowering(flydsl_call_p, _lowering, platform="rocm")


def flydsl_call(
    *array_args: jax.Array,
    kernel,
    out_shape,
    scalars: dict[str, int | float] | None = None,
    output_positions: tuple[int, ...] | list[int] | None = None,
    input_output_aliases: dict[int, int] | None = None,
    compile_hints: dict[str, Any] | None = None,
):
    """Call one FlyDSL launcher from JAX.

    Tensor parameters are supplied by ``array_args`` and ``out_shape``. Scalar
    parameters are static call attributes. The launcher must have one final
    ``fx.Stream`` parameter. ``compile_hints`` is applied as FlyDSL
    ``CompilationContext.compile_hints`` while tracing and compiling; it must
    be JSON-serializable.
    """
    _ensure_gpu_initialized()
    _patch_flydsl_rocm_toolkit()
    ensure_target_registered()
    layout = _inspect_launcher(kernel)

    flat_outputs, output_tree = jax.tree_util.tree_flatten(out_shape)
    arrays = tuple(jnp.asarray(array) for array in array_args)
    num_outputs = len(flat_outputs)
    input_output_aliases = input_output_aliases or {}

    buffer_positions = [
        index for index, kind in enumerate(layout.kinds) if kind == "buffer"
    ]
    if output_positions is None:
        output_positions = tuple(buffer_positions[-num_outputs:])
    else:
        output_positions = tuple(int(index) for index in output_positions)

    if len(output_positions) != num_outputs:
        raise ValueError(
            f"Expected {num_outputs} output positions, got {len(output_positions)}"
        )
    if len(set(output_positions)) != len(output_positions):
        raise ValueError("FlyDSL output positions must be unique")
    invalid_output_positions = set(output_positions) - set(buffer_positions)
    if invalid_output_positions:
        raise ValueError(
            "FlyDSL output positions must name buffer parameters, got "
            f"{sorted(invalid_output_positions)}"
        )
    output_positions = tuple(sorted(output_positions))
    if len(arrays) + num_outputs - len(input_output_aliases) != layout.num_buffers:
        raise ValueError(
            f"Launcher has {layout.num_buffers} buffers, but got "
            f"{len(arrays)} inputs, {num_outputs} outputs, and "
            f"{len(input_output_aliases)} aliases"
        )
    for input_index, output_index in input_output_aliases.items():
        if input_index < 0 or input_index >= len(arrays):
            raise ValueError(f"Aliased input index {input_index} is out of range")
        if output_index < 0 or output_index >= num_outputs:
            raise ValueError(f"Aliased output index {output_index} is out of range")
        output = flat_outputs[output_index]
        if arrays[input_index].shape != output.shape or arrays[input_index].dtype != output.dtype:
            raise ValueError(
                "Aliased input and output must have identical shape and dtype"
            )
    if len(set(input_output_aliases.values())) != len(input_output_aliases):
        raise ValueError("Each FlyDSL output can alias at most one input")

    scalar_types = layout.scalar_type_by_name
    scalar_names = tuple(
        name for name, kind in zip(layout.names, layout.kinds) if kind == "scalar"
    )
    scalars = scalars or {}
    missing = set(scalar_names) - set(scalars)
    extra = set(scalars) - set(scalar_names)
    if missing or extra:
        raise ValueError(
            f"FlyDSL scalar mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    scalar_pairs = tuple(
        (
            name,
            float(scalars[name])
            if scalar_types[name].startswith("Float")
            else int(scalars[name]),
        )
        for name in scalar_names
    )

    input_specs = tuple((tuple(array.shape), array.dtype) for array in arrays)
    output_specs = tuple(
        (tuple(shape.shape), np.dtype(shape.dtype)) for shape in flat_outputs
    )
    results = flydsl_call_p.bind(
        *arrays,
        kernel=kernel,
        layout=layout,
        output_positions=output_positions,
        scalar_pairs=scalar_pairs,
        input_specs=input_specs,
        output_specs=output_specs,
        out_shapes=tuple(flat_outputs),
        input_output_aliases=tuple(sorted(input_output_aliases.items())),
        compile_hints_json=(
            json.dumps(compile_hints, sort_keys=True) if compile_hints else ""
        ),
    )
    return jax.tree_util.tree_unflatten(output_tree, results)
