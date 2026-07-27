# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only plumbing tests for the guarded MXFP4 offset prototype."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest


g = importlib.import_module("jax_aiter.gemm_fp4.gemm_fp4")
ops = importlib.import_module("jax_aiter.ops.gemm_fp4")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("off", 0), ("auto", 1), ("force64", 2)],
)
def test_offset_mode_parser_has_stable_integer_values(raw, expected):
    assert g._parse_mxfp4_offset_mode(raw) == expected


def test_offset_mode_defaults_to_off(monkeypatch):
    monkeypatch.delenv("JA_MXFP4_OFFSET_MODE", raising=False)
    assert g._parse_mxfp4_offset_mode() == 0


@pytest.mark.parametrize("raw", ["", "on", "AUTO", "force", "1"])
def test_offset_mode_unknown_value_fails_closed(raw):
    with pytest.raises(ValueError, match="JA_MXFP4_OFFSET_MODE"):
        g._parse_mxfp4_offset_mode(raw)


def test_activation_and_gradient_dual_attrs_use_mode_but_weight_stays_off(
    monkeypatch,
):
    calls = []

    def fake_dual(_x, **attrs):
        calls.append(attrs)
        return object(), object(), object(), object()

    monkeypatch.setattr(g, "_cast_mxfp4_dual_op", fake_dual)
    monkeypatch.setattr(g, "_MXFP4_OFFSET_MODE", 1)

    g._cast_act_dual_raw(object())
    g._cast_grad_dual_raw(object())
    g._cast_wt_dual_raw(object())
    g._cast_wt_colwise_unshuf_raw(object())

    assert [int(call["offset_mode"]) for call in calls] == [1, 1, 0, 0]


def test_generic_low_level_casts_default_backend_offset_attr_off(monkeypatch):
    calls = []

    def fake_ffi_call(target, _out_shapes, **_kwargs):
        def invoke(_x, **attrs):
            calls.append((target, attrs))
            return object()

        return invoke

    monkeypatch.setattr(ops, "_ensure_cast_registered", lambda: None)
    monkeypatch.setattr(ops, "_ensure_dual_registered", lambda: None)
    monkeypatch.setattr(ops.jax.ffi, "ffi_call", fake_ffi_call)

    class FakeArray:
        shape = (32, 64)

    ops.cast_mxfp4(FakeArray(), shuffle_fp4=False)
    ops.cast_mxfp4_dual(FakeArray(), shuffle_fp4=False)

    assert [target for target, _attrs in calls] == [
        "CastMxfp4JA",
        "CastMxfp4DualJA",
    ]
    assert [int(attrs["offset_mode"]) for _target, attrs in calls] == [0, 0]
    assert all(
        isinstance(attrs["offset_mode"], np.int32) for _target, attrs in calls
    )


def test_low_level_dual_cast_records_explicit_integer_backend_mode(monkeypatch):
    captured = {}

    def fake_ffi_call(_target, _out_shapes, **_kwargs):
        def invoke(_x, **attrs):
            captured.update(attrs)
            return object()

        return invoke

    monkeypatch.setattr(ops, "_ensure_dual_registered", lambda: None)
    monkeypatch.setattr(ops.jax.ffi, "ffi_call", fake_ffi_call)

    class FakeArray:
        shape = (32, 64)

    ops.cast_mxfp4_dual(FakeArray(), shuffle_fp4=False, offset_mode=2)
    assert captured["offset_mode"] == np.int32(2)
    assert isinstance(captured["offset_mode"], np.int32)


def test_ffi_and_kernel_source_expose_only_two_typed_u32_specializations():
    repo = Path(__file__).resolve().parents[1]
    makefile = (repo / "Makefile").read_text()
    bridge = (
        repo / "csrc" / "ffi" / "cast_mxfp4" / "cast_mxfp4_ja.cu"
    ).read_text()
    kernel = (
        repo
        / "csrc"
        / "ffi"
        / "cast_mxfp4"
        / "cast_transpose_mxfp4_kernel_shuffled.cu"
    ).read_text()
    normalized = " ".join(kernel.split())

    assert bridge.count('.Attr<int>("offset_mode")') == 2
    assert "CAST_MXFP4_GUARD :=" in makefile
    assert (
        "$(CAST_MXFP4_SRC) $(CAST_MXFP4_KERNEL) $(CAST_MXFP4_GUARD)"
        in makefile
    )
    assert "template<typename OffsetT," in normalized
    assert normalized.count("LAUNCH_TYPED_KERNEL(uint32_t,") == 2
    assert (
        "LAUNCH_TYPED_KERNEL(uint32_t, true, true, true, true, true, "
        "false, true, false, false)"
    ) in normalized
    assert (
        "LAUNCH_TYPED_KERNEL(uint32_t, true, true, true, true, true, "
        "false, false, false, false)"
    ) in normalized
    assert "LAUNCH_TYPED_KERNEL(int64_t," in normalized


def _load_task4_parity_script():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "mxfp4_cast_byte_parity.py"
    )
    spec = importlib.util.spec_from_file_location("mxfp4_cast_byte_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task4_parity_inventory_and_edge_values_are_exact():
    parity = _load_task4_parity_script()
    cells = [
        (cell["role"], cell["M"], cell["K"])
        for cell in parity.TASK4_PRODUCTION_CELLS
    ]
    assert cells == [
        ("activation", 32768, 4096),
        ("activation", 32768, 14336),
        ("gradient", 32768, 1024),
        ("gradient", 32768, 4096),
        ("gradient", 32768, 14336),
    ]
    assert parity.TASK4_MODE_VALUES == {
        "off_u64": 0,
        "auto_u32": 1,
        "force64_u64": 2,
    }
    edge = parity.task4_edge_values()
    assert np.any(edge == 0.0)
    assert np.any((edge == 0.0) & np.signbit(edge))
    assert np.any(np.isposinf(edge))
    assert np.any(np.isneginf(edge))
    assert np.any(np.isnan(edge))
    finite_nonzero = np.abs(edge[np.isfinite(edge) & (edge != 0)])
    assert np.all(np.log2(finite_nonzero) == np.floor(np.log2(finite_nonzero)))


def test_task4_parity_comparator_requires_all_four_exact_outputs():
    parity = _load_task4_parity_script()
    arrays = ("row_fp4", "col_fp4", "row_scale", "col_scale")
    hashes = {name: f"hash-{name}" for name in arrays}
    records = {
        "activation_m32768_k4096": {
            mode: {"hashes": dict(hashes)}
            for mode in parity.TASK4_MODE_VALUES
        }
    }
    summary = parity.compare_task4_records(records)
    assert summary["status"] == "PASS"
    assert summary["cells_checked"] == 1

    records["activation_m32768_k4096"]["auto_u32"]["hashes"][
        "col_scale"
    ] = "different"
    with pytest.raises(AssertionError, match="col_scale"):
        parity.compare_task4_records(records)


def test_task4_hlo_mode_parser_accepts_compiled_backend_config_format():
    parity = _load_task4_parity_script()
    hlo = (
        'custom_call_target="CastMxfp4DualJA", '
        "backend_config={offset_mode = 0 : i32, scale_mode = 0 : i32}"
    )
    assert parity._hlo_backend_modes(hlo) == [0]


def test_task4_shader_log_parser_uses_exact_observed_shader_name():
    parity = _load_task4_parity_script()
    expected = parity._task4_shader_name("gradient", "u32")
    log = (
        ":3:rocvirtual.cpp :3596: 1 us: ShaderName : unrelated\n"
        f":3:rocvirtual.cpp :3596: 2 us: ShaderName : {expected}\n"
    )

    parsed = parity.parse_task4_shader_log(log, expected)

    assert parsed["observed_shader_name"] == expected
    assert parsed["kernel_offset_type"] == "uint32_t"
    assert parsed["matching_launch_count"] == 1


def test_task4_shader_log_parser_fails_closed_on_gradient_u64_fallback():
    parity = _load_task4_parity_script()
    expected_u32 = parity._task4_shader_name("gradient", "u32")
    observed_u64 = parity._task4_shader_name("gradient", "u64")
    log = (
        ":3:rocvirtual.cpp :3596: 1 us: "
        f"ShaderName : {observed_u64}\n"
    )

    with pytest.raises(AssertionError, match="exact expected ShaderName"):
        parity.parse_task4_shader_log(log, expected_u32)


def test_task4_shader_log_parser_rejects_ambiguous_cast_launches():
    parity = _load_task4_parity_script()
    expected = parity._task4_shader_name("activation", "u32")
    unexpected = parity._task4_shader_name("activation", "u64")
    log = (
        f":3:rocvirtual.cpp :3596: 1 us: ShaderName : {expected}\n"
        f":3:rocvirtual.cpp :3596: 2 us: ShaderName : {unexpected}\n"
    )

    with pytest.raises(AssertionError, match="unexpected MXFP4 cast launch"):
        parity.parse_task4_shader_log(log, expected)


def test_pure_host_offset_guard_binary(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source = repo / "tests" / "mxfp4_offset_guard_host_test.cc"
    binary = tmp_path / "mxfp4_offset_guard_host_test"
    compile_result = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(repo / "csrc" / "ffi" / "cast_mxfp4"),
            str(source),
            "-o",
            str(binary),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        [str(binary)], text=True, capture_output=True, check=False
    )
    assert run_result.returncode == 0, run_result.stderr
