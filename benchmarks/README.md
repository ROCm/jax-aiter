# JAX-AITER Benchmarks

Performance benchmarking scripts for JAX-AITER.

## benchmark_mha.py

Compares Multi-Head Attention between:
- **Pure JAX**: Uses `jax_aiter.baseline.mha_attn.attention_ref`
- **JAX-AITER**: Uses `jax_aiter.mha.flash_attn_func`

**Key details:**
- BSHD layout (batch, seq_len, num_heads, head_dim)
- Default dtype: bfloat16
- 22 configurations with various batch sizes, sequence lengths (up to 8192), and head dimensions (64-256)
- Supports both causal and non-causal attention
- Each config runs 10 times by default, reports median timings

**Usage:**
```bash
cd jax-aiter
python benchmarks/benchmark_mha.py
```

**Output:**
- Per-run timing details
- Per-configuration median results
- Overall median and mean speedup across all configurations

**Optional CSV export:**
Add this line at the end of `main()` to export results:
```python
export_results_csv(results, filename="benchmark_results.csv")
```

**Requirements:**
- JAX with GPU support
- numpy, pandas

## Notes

- Performance varies based on hardware and system configuration
- Run benchmarks in a controlled environment for consistent results
