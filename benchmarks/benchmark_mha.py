# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
"""
Performance comparison between jax-aiter and pure JAX attention implementations.
Focuses on BSHD layout with various hyperparameter configurations.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom

import jax_aiter
from jax_aiter.mha import flash_attn_func as jax_flash_attn_func
from jax_aiter.baseline.mha_attn import attention_ref


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int
    causal: bool = False
    dtype: str = "bf16"
    extra_batch_dims: Tuple[int, ...] = ()

    def __str__(self):
        if self.extra_batch_dims:
            extra_str = "x".join(map(str, self.extra_batch_dims)) + "x"
            return f"B{extra_str}{self.batch_size}_S{self.seq_len}_H{self.num_heads}_D{self.head_dim}"
        return f"B{self.batch_size}_S{self.seq_len}_H{self.num_heads}_D{self.head_dim}"


def get_benchmark_configs() -> List[BenchmarkConfig]:
    """Get configurations focusing on large head dimensions (skip 128)."""
    configs = [
        # Smaller sequences with various head dims.
        BenchmarkConfig(2, 1024, 8, 64),
        BenchmarkConfig(2, 1024, 8, 192),
        BenchmarkConfig(2, 1024, 8, 224),
        BenchmarkConfig(2, 1024, 8, 256),
        # Medium sequences.
        BenchmarkConfig(1, 2048, 8, 192),
        BenchmarkConfig(2, 2048, 16, 192),
        BenchmarkConfig(1, 2048, 8, 256),
        BenchmarkConfig(4, 2048, 8, 224),
        # Large sequences with large head dims.
        BenchmarkConfig(1, 4096, 8, 192),
        BenchmarkConfig(2, 4096, 16, 192),
        BenchmarkConfig(1, 4096, 8, 224),
        BenchmarkConfig(2, 4096, 8, 256),
        BenchmarkConfig(4, 4096, 32, 64),
        # Extra large sequences.
        BenchmarkConfig(1, 8192, 8, 64),
        BenchmarkConfig(1, 8192, 8, 192),
        BenchmarkConfig(1, 8192, 8, 224),
        BenchmarkConfig(1, 8192, 8, 256),
        # Causal attention tests with large dims.
        BenchmarkConfig(2, 2048, 8, 192, causal=True),
        BenchmarkConfig(1, 4096, 8, 256, causal=True),
        BenchmarkConfig(2, 4096, 16, 192, causal=True),
        # Various batch sizes and heads.
        BenchmarkConfig(4, 2048, 32, 192),
        BenchmarkConfig(8, 1024, 4, 256),
    ]
    return configs


def generate_inputs(
    config: BenchmarkConfig, key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate Q, K, V tensors for benchmarking using JAX random."""
    dtype_map = {
        "fp16": jnp.float16,
        "fp32": jnp.float32,
        "bf16": jnp.bfloat16,
    }
    dtype = dtype_map.get(config.dtype, jnp.bfloat16)

    key_q, key_k, key_v = jrandom.split(key, 3)

    # Shape includes any extra batch dimensions.
    shape = (
        *config.extra_batch_dims,
        config.batch_size,
        config.seq_len,
        config.num_heads,
        config.head_dim,
    )

    q = jrandom.normal(key_q, shape, dtype=dtype)
    k = jrandom.normal(key_k, shape, dtype=dtype)
    v = jrandom.normal(key_v, shape, dtype=dtype)

    return q, k, v


def _flatten_to_bshd(tensor: jnp.ndarray) -> jnp.ndarray:
    """Flatten all leading dimensions into batch dimension to get BSHD format."""
    if tensor.ndim == 4:
        return tensor
    # Flatten all dimensions except the last 3 (S, H, D).
    shape = tensor.shape
    effective_batch = int(np.prod(shape[:-3]))
    return tensor.reshape(effective_batch, shape[-3], shape[-2], shape[-1])


def _unflatten_from_bshd(
    tensor: jnp.ndarray, original_shape: Tuple[int, ...]
) -> jnp.ndarray:
    """Unflatten batch dimension back to original leading dimensions."""
    if len(original_shape) == 4:
        return tensor
    # Restore original leading dimensions.
    return tensor.reshape(*original_shape[:-3], *tensor.shape[-3:])


def pure_jax_attention(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, *, causal: bool, upcast: bool = True
) -> jnp.ndarray:
    """Reference attention implementation using attention_ref from baseline."""
    original_shape = q.shape

    # Flatten to BSHD for computation.
    q_flat = _flatten_to_bshd(q)
    k_flat = _flatten_to_bshd(k)
    v_flat = _flatten_to_bshd(v)

    # Use attention_ref from baseline with window_size=(-1, -1) for no windowing.
    out_flat, _, _ = attention_ref(
        q_flat,
        k_flat,
        v_flat,
        query_padding_mask=None,
        key_padding_mask=None,
        attn_bias=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=causal,
        window_size=(-1, -1),
        softcap=0.0,
        upcast=upcast,
        reorder_ops=False,
        key_leftpad=None,
    )

    # Unflatten back to original shape.
    return _unflatten_from_bshd(out_flat, original_shape)


def jax_aiter_attention(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, *, causal: bool
) -> jnp.ndarray:
    """JAX-AITER flash attention wrapper returning only the output tensor."""
    original_shape = q.shape

    # Flatten to BSHD for computation.
    q_flat = _flatten_to_bshd(q)
    k_flat = _flatten_to_bshd(k)
    v_flat = _flatten_to_bshd(v)

    result = jax_flash_attn_func(
        q_flat,
        k_flat,
        v_flat,
        dropout_p=0.0,
        causal=causal,
        window_size=(-1, -1),
        bias=None,
        alibi_slopes=None,
        deterministic=False,
        return_lse=False,
        return_attn_probs=False,
    )

    if isinstance(result, tuple):
        out_flat = result[0]
    else:
        out_flat = result

    # Unflatten back to original shape.
    return _unflatten_from_bshd(out_flat, original_shape)


def benchmark_attention(
    config: BenchmarkConfig,
    key: jax.Array,
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
) -> Dict[str, float]:
    """Benchmark both implementations and return timing results."""
    q, k, v = generate_inputs(config, key)

    pure_jax_fn = jax.jit(
        lambda q_, k_, v_: pure_jax_attention(q_, k_, v_, causal=config.causal)
    )
    jax_aiter_fn = jax.jit(
        lambda q_, k_, v_: jax_aiter_attention(q_, k_, v_, causal=config.causal)
    )

    # Warmup for pure JAX baseline.
    for _ in range(warmup_runs):
        pure_jax_fn(q, k, v).block_until_ready()

    # Benchmark pure JAX baseline.
    start = time.perf_counter()
    for _ in range(benchmark_runs):
        pure_jax_fn(q, k, v).block_until_ready()
    pure_jax_time = (time.perf_counter() - start) / benchmark_runs * 1000  # ms

    # Warmup for JAX-AITER.
    for _ in range(warmup_runs):
        jax_aiter_fn(q, k, v).block_until_ready()

    # Benchmark JAX-AITER.
    start = time.perf_counter()
    for _ in range(benchmark_runs):
        jax_aiter_fn(q, k, v).block_until_ready()
    jax_aiter_time = (time.perf_counter() - start) / benchmark_runs * 1000  # ms

    # Calculate the speedup.
    speedup = pure_jax_time / jax_aiter_time

    return {
        "pure_jax_ms": pure_jax_time,
        "jax_aiter_ms": jax_aiter_time,
        "speedup": speedup,
    }


def print_results_table(results: List[Dict[str, Any]]):
    """Print results in a formatted table."""
    if not results:
        print("No successful benchmark runs to report.")
        return

    df = pd.DataFrame(results)
    avg_speedup = float(np.mean(df["speedup"].to_numpy()))

    df_fmt = df.copy()
    df_fmt["pure_jax_ms"] = df_fmt["pure_jax_ms"].apply(lambda x: f"{x:.3f}")
    df_fmt["jax_aiter_ms"] = df_fmt["jax_aiter_ms"].apply(lambda x: f"{x:.3f}")
    df_fmt["speedup"] = df_fmt["speedup"].apply(lambda x: f"{x:.2f}x")

    print("\n" + "=" * 100)
    print("Performance Comparison: Pure JAX vs JAX-AITER")
    print("=" * 100)
    print("\nConfiguration Details:")
    print(df_fmt.to_string(index=False))

    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    print("=" * 100)


def print_results_table_with_stats(results: List[Dict[str, Any]]):
    """Print results in a formatted table with statistical information."""
    if not results:
        print("No successful benchmark runs to report.")
        return

    df = pd.DataFrame(results)
    median_speedup = float(np.median(df["speedup"].to_numpy()))
    mean_speedup = float(np.mean(df["speedup"].to_numpy()))

    df_fmt = df.copy()
    df_fmt["pure_jax_ms"] = df_fmt["pure_jax_ms"].apply(lambda x: f"{x:.3f}")
    df_fmt["jax_aiter_ms"] = df_fmt["jax_aiter_ms"].apply(lambda x: f"{x:.3f}")
    df_fmt["speedup"] = df_fmt["speedup"].apply(lambda x: f"{x:.2f}x")

    columns_to_drop = ["num_successful_runs", "extra_batch_dims", "effective_batch"]
    display_df = df_fmt.drop(columns=columns_to_drop, errors="ignore")

    print("\n" + "=" * 100)
    print("Performance Comparison: Pure JAX vs JAX-AITER (Median values from 10 runs)")
    print("=" * 100)
    print("\nConfiguration Details:")
    print(display_df.to_string(index=False))

    print(f"\nMedian Speedup: {median_speedup:.2f}x")
    print(f"Mean Speedup: {mean_speedup:.2f}x")
    print("=" * 100)


def export_results_csv(
    results: List[Dict[str, Any]], filename: str = "benchmark_results.csv"
):
    """Export results to CSV for plotting."""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nResults exported to: {filename}")


def main(num_runs: int = 10):
    """Run benchmarks and report results.

    Args:
        num_runs: Number of times to run each benchmark for median calculation
    """
    print("JAX-AITER vs Pure JAX Performance Benchmark with Higher Dimensions Support")
    print(f"Device: {jax.devices()[0]}")
    print(f"JAX version: {jax.__version__}")
    print(f"Number of runs per config: {num_runs} (reporting median values)")
    print("")

    configs = get_benchmark_configs()
    results = []
    key = jrandom.PRNGKey(0)

    for config in configs:
        print(f"Benchmarking: {config}")

        pure_jax_times = []
        jax_aiter_times = []
        speedups = []

        for run in range(num_runs):
            key, subkey = jrandom.split(key)

            try:
                timings = benchmark_attention(config, subkey)
                pure_jax_times.append(timings["pure_jax_ms"])
                jax_aiter_times.append(timings["jax_aiter_ms"])
                speedups.append(timings["speedup"])

                print(
                    f"  Run {run+1}/{num_runs}: Pure JAX: {timings['pure_jax_ms']:.3f} ms, JAX-AITER: {timings['jax_aiter_ms']:.3f} ms, Speedup: {timings['speedup']:.2f}x"
                )

            except Exception as e:
                print(f"  Run {run+1}/{num_runs} failed: {str(e)}")

        if pure_jax_times:
            median_pure_jax = np.median(pure_jax_times)
            median_jax_aiter = np.median(jax_aiter_times)
            median_speedup = np.median(speedups)

            effective_batch = (
                int(np.prod(config.extra_batch_dims)) * config.batch_size
                if config.extra_batch_dims
                else config.batch_size
            )

            result = {
                "batch_size": config.batch_size,
                "seq_len": config.seq_len,
                "num_heads": config.num_heads,
                "head_dim": config.head_dim,
                "causal": config.causal,
                "dtype": config.dtype,
                "extra_batch_dims": config.extra_batch_dims
                if config.extra_batch_dims
                else "",
                "effective_batch": effective_batch,
                "pure_jax_ms": median_pure_jax,
                "jax_aiter_ms": median_jax_aiter,
                "speedup": median_speedup,
                "num_successful_runs": len(pure_jax_times),
            }
            results.append(result)

            print(
                f"  MEDIAN: Pure JAX: {median_pure_jax:.3f} ms, JAX-AITER: {median_jax_aiter:.3f} ms, Speedup: {median_speedup:.2f}x"
            )
            if config.extra_batch_dims:
                print(
                    f"  Extra dims: {config.extra_batch_dims}, Effective batch: {effective_batch}"
                )
            print("")

    print_results_table_with_stats(results)


if __name__ == "__main__":
    main()
