#!/usr/bin/env bash
# CPU-only recipe/parser tests; no JAX, Docker, or GPU process is started.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PERF_RUNNER="$ROOT/scripts/recipes/run_nvfp4_match_8b.sh"
TTC_RUNNER="$ROOT/scripts/recipes/run_mlperf_ttc_8b.sh"
LOG_VALIDATOR="$ROOT/scripts/validate_maxtext_train_log.py"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

count_exact() {
  local expected="$1" file="$2"
  awk -v expected="$expected" '$0 == expected { count++ } END { print count + 0 }' "$file"
}

count_prefix() {
  local prefix="$1" file="$2"
  awk -v prefix="$prefix" 'index($0, prefix) == 1 { count++ } END { print count + 0 }' "$file"
}

assert_line() {
  local expected="$1" file="$2" count
  count="$(count_exact "$expected" "$file")"
  [[ "$count" == 1 ]] || {
    printf 'expected one line %q in %s, got %s\n' "$expected" "$file" "$count" >&2
    return 1
  }
}

assert_text() {
  local expected="$1" file="$2"
  awk -v expected="$expected" 'index($0, expected) { found=1 } END { exit !found }' "$file" || {
    printf 'expected text %q in %s\n' "$expected" "$file" >&2
    return 1
  }
}

assert_absent() {
  local unexpected="$1" file="$2"
  ! awk -v unexpected="$unexpected" 'index($0, unexpected) { found=1 } END { exit !found }' "$file" || {
    printf 'unexpected text %q in %s\n' "$unexpected" "$file" >&2
    return 1
  }
}

assert_one_quantization_arg() {
  local file="$1" count
  count="$(count_prefix "RESOLVED_ARG=quantization=" "$file")"
  [[ "$count" == 1 ]] || {
    printf 'expected one quantization arg in %s, got %s\n' "$file" "$count" >&2
    return 1
  }
}

clean_env() {
  env -i PATH="$PATH" HOME="$TMP" RECIPE_DRY_RUN=1 \
    PROJECT_ROOT=/opt/mxfp4-repro PYTHON_BIN=/not-executed/python3 "$@"
}

run_perf() {
  local mode="$1" capture="$2"
  clean_env bash "$PERF_RUNNER" "$mode" /outputs 50 >"$capture" 2>&1
}

run_ttc() {
  local mode="$1" capture="$2"
  clean_env bash "$TTC_RUNNER" "${mode}_ttc" "$mode" /outputs 6000 >"$capture" 2>&1
}

check_perf_common() {
  local capture="$1"
  assert_text "autotune=4 weight_dtype=bfloat16 mu_dtype=bfloat16" "$capture" &&
    assert_text "batch=9 global_batch=72 sequence=8192 init_seed=0 mem_fraction=.97" "$capture" &&
    assert_line "RESOLVED_ARG=weight_dtype=bfloat16" "$capture" &&
    assert_line "RESOLVED_ARG=mu_dtype=bfloat16" "$capture" &&
    assert_line "RESOLVED_ARG=per_device_batch_size=9" "$capture" &&
    assert_line "RESOLVED_ARG=global_batch_size_to_train_on=72" "$capture" &&
    assert_line "RESOLVED_ARG=init_weights_seed=0" "$capture" &&
    assert_line "RESOLVED_ARG=scan_layers=False" "$capture" &&
    assert_line "RESOLVED_ARG=enable_nnx=False" "$capture" &&
    assert_line "RESOLVED_ARG=pure_nnx=False" "$capture" &&
    assert_line "RESOLVED_ARG=pure_nnx_decoder=False" "$capture" &&
    assert_text "project_root=/opt/mxfp4-repro jax_aiter_root=/opt/mxfp4-repro/jax-aiter" "$capture" &&
    assert_text "maxtext_root=/opt/mxfp4-repro/maxtext" "$capture" &&
    assert_text "queue_interposition:0,register_enabled:0" "$capture" &&
    assert_text "dev_kernarg:1,fine_grain_pcie:1" "$capture" &&
    assert_text "rocm_sdk_library_path=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib:/usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries/lib" "$capture" &&
    assert_absent "collective_network=" "$capture" &&
    assert_text "--xla_gpu_autotune_level=4" "$capture" &&
    assert_one_quantization_arg "$capture"
}

check_ttc_common() {
  local capture="$1"
  assert_text "autotune=5 weight_dtype=float32 mu_dtype=float32 use_iota_embed=False batch=4 global_batch=32 sequence=8192 init_seed=0 mem_fraction=.97" "$capture" &&
    assert_line "RESOLVED_ARG=weight_dtype=float32" "$capture" &&
    assert_line "RESOLVED_ARG=mu_dtype=float32" "$capture" &&
    assert_line "RESOLVED_ARG=use_iota_embed=False" "$capture" &&
    assert_line "RESOLVED_ARG=global_batch_size_to_train_on=32" "$capture" &&
    assert_line "RESOLVED_ARG=scan_layers=False" "$capture" &&
    assert_one_quantization_arg "$capture"
}

test_perf_mxfp4_direct_mha() {
  local capture="$TMP/perf-mxfp4"
  run_perf mxfp4 "$capture" &&
    check_perf_common "$capture" &&
    assert_text "recipe=blog_batch9 mode=mxfp4 label=MXFP4 model=llama3.1-8b model_controls=llama31_mlperf processes=1 steps=50 measurement_window=completed_steps_40_49" "$capture" &&
    assert_line "RESOLVED_ARG=quantization=aiter_fp4" "$capture" &&
    assert_line "RESOLVED_ARG=attention=aiter_flash" "$capture" &&
    assert_text "runtime=source" "$capture" &&
    assert_line "RESOLVED_ARG=use_jax_aiter=True" "$capture" &&
    assert_line "RESOLVED_ARG=use_iota_embed=False" "$capture" &&
    assert_line "RESOLVED_ARG=remat_policy=minimal_flash_save_fp4_wtcol" "$capture" &&
    assert_text "scheduler=nvidia combine_threshold=67108864" "$capture" &&
    assert_text "no_scratch_reclaim:1" "$capture" &&
    assert_text "hadamard_passes=wgrad sr_passes=wgrad_col dgrad_partition=gather_packed dgrad_reuse_fwd_col=1 remat_save_col=wt pack_gateup_ag=1" "$capture" &&
    assert_text "fp4_select=dispatch fp4_attention_gemm=1 fp4_mlp_gemm=1 bf16_hipblaslt_fallback=0 sr_key_mode=maxtext_runtime_params_rng" "$capture" &&
    assert_text "ja_mha_atomic_fp32=0 ja_mha_bf16_cvt=2" "$capture" &&
    assert_absent "RESOLVED_ARG=aiter_attention=False" "$capture"
}

test_perf_plain_fp8() {
  local capture="$TMP/perf-fp8"
  run_perf fp8 "$capture" &&
    check_perf_common "$capture" &&
    assert_text "mode=fp8 label=Plain FP8 model=llama3.1-8b model_controls=maxtext_defaults processes=1 steps=50" "$capture" &&
    assert_line "RESOLVED_ARG=quantization=fp8" "$capture" &&
    assert_line "RESOLVED_ARG=attention=cudnn_flash_te" "$capture" &&
    assert_line "RESOLVED_ARG=use_jax_aiter=False" "$capture" &&
    assert_line "RESOLVED_ARG=use_iota_embed=True" "$capture" &&
    assert_line "RESOLVED_ARG=remat_policy=minimal_flash" "$capture" &&
    assert_text "scheduler=utd combine_threshold=8589934592" "$capture" &&
    assert_text "no_scratch_reclaim:unset" "$capture" &&
    assert_text "te_atomic_fp32=0 te_bf16_cvt=2" "$capture" &&
    assert_absent "RESOLVED_ARG=query_pre_attn_scalar=" "$capture"
}

test_perf_bf16_no_quantization() {
  local capture="$TMP/perf-bf16"
  run_perf bf16 "$capture" &&
    check_perf_common "$capture" &&
    assert_text "mode=bf16 label=BF16 model=llama3.1-8b model_controls=maxtext_defaults processes=1 steps=50" "$capture" &&
    assert_line "RESOLVED_ARG=quantization=" "$capture" &&
    assert_line "RESOLVED_ARG=attention=cudnn_flash_te" "$capture" &&
    assert_line "RESOLVED_ARG=use_iota_embed=True" "$capture" &&
    assert_line "RESOLVED_ARG=remat_policy=minimal" "$capture" &&
    assert_text "scheduler=utd combine_threshold=8589934592" "$capture" &&
    assert_text "no_scratch_reclaim:unset" "$capture" &&
    assert_text "quantization=none attention=cudnn_flash_te" "$capture" &&
    assert_absent "RESOLVED_ARG=query_pre_attn_scalar=" "$capture"
}

test_convergence_modes() {
  local mode capture quant label attention
  for mode in mxfp4 te_fp8_currentscaling bf16; do
    capture="$TMP/ttc-$mode"
    case "$mode" in
      mxfp4) quant=aiter_fp4; label=MXFP4; attention=aiter_flash ;;
      te_fp8_currentscaling) quant=te_fp8_currentscaling; label="TE FP8 current scaling"; attention=cudnn_flash_te ;;
      bf16) quant=""; label=BF16; attention=cudnn_flash_te ;;
    esac
    run_ttc "$mode" "$capture" &&
      check_ttc_common "$capture" &&
      assert_text "runner=convergence tag=${mode}_ttc mode=${mode} label=${label} processes=1 steps=6000" "$capture" &&
      assert_line "RESOLVED_ARG=quantization=$quant" "$capture" &&
      assert_line "RESOLVED_ARG=attention=$attention" "$capture" &&
      assert_line "RESOLVED_ARG=enable_mlperf_logging=True" "$capture" || return 1
  done
}

test_rejects_stale_aliases() {
  local alias capture
  for alias in fp8_aiter_mha te_fp8 te_fp8_currentscaling mxfp4_mlponly; do
    capture="$TMP/reject-$alias"
    if run_perf "$alias" "$capture"; then
      printf 'stale alias %s unexpectedly succeeded\n' "$alias" >&2
      return 1
    fi
  done
}

test_rejects_noncanonical_mem_fraction() {
  local capture="$TMP/bad-memfrac"
  if env -i PATH="$PATH" HOME="$TMP" RECIPE_DRY_RUN=1 \
      XLA_PYTHON_CLIENT_MEM_FRACTION=.90 \
      bash "$PERF_RUNNER" mxfp4 /outputs 50 >"$capture" 2>&1; then
    printf 'noncanonical mem fraction unexpectedly succeeded\n' >&2
    return 1
  fi
  assert_text "canonical memory fraction is .97" "$capture"
}

test_llama31_model_controls_are_always_explicit() {
  local capture="$TMP/perf-llama31"
  run_perf mxfp4 "$capture" &&
    assert_line "RESOLVED_ARG=query_pre_attn_scalar=0.08838834764831843" "$capture" &&
    assert_line "RESOLVED_ARG=rope_use_scale=False" "$capture" &&
    assert_line "RESOLVED_ARG=normalize_embedding_logits=False" "$capture" &&
    assert_line "RESOLVED_ARG=megatron_init_std=0.02" "$capture" &&
    assert_line "RESOLVED_ARG=megatron_residual_scale=True" "$capture" &&
    assert_line "RESOLVED_ARG=num_vocab_tiling=1" "$capture"
}

test_blog_profile_rejects_recipe_drift() {
  local capture="$TMP/perf-drift"
  if clean_env env PER_DEVICE_BATCH=4 \
      bash "$PERF_RUNNER" mxfp4 /outputs 50 >"$capture" 2>&1; then
    printf 'blog profile accepted a batch override\n' >&2
    return 1
  fi
  assert_text "PER_DEVICE_BATCH must be 9 for blog_batch9" "$capture"
}

test_ci_profile_accepts_explicit_regression_point() {
  local capture="$TMP/perf-ci"
  clean_env env RECIPE_PROFILE=ci_regression ICI_FSDP_PARALLELISM=4 \
    PER_DEVICE_BATCH=4 GLOBAL_BATCH_SIZE=16 WEIGHT_DTYPE=float32 \
    MU_DTYPE=float32 AUTOTUNE_LEVEL=5 REMAT_POLICY=minimal_flash_save_fp4col \
    JA_FP4_REMAT_SAVE_COL=both \
    bash "$PERF_RUNNER" mxfp4 /outputs 50 >"$capture" 2>&1 &&
    assert_text "recipe=ci_regression mode=mxfp4" "$capture" &&
    assert_line "RESOLVED_ARG=per_device_batch_size=4" "$capture" &&
    assert_line "RESOLVED_ARG=global_batch_size_to_train_on=16" "$capture" &&
    assert_line "RESOLVED_ARG=weight_dtype=float32" "$capture" &&
    assert_text "remat_save_col=both" "$capture"
}

test_installed_wheel_runtime_resolution() {
  local capture="$TMP/perf-installed-runtime"
  clean_env env JAX_AITER_RUNTIME=installed \
    bash "$PERF_RUNNER" mxfp4 /outputs 50 >"$capture" 2>&1 &&
    assert_text "runtime=installed" "$capture" &&
    assert_line "RESOLVED_ARG=attention=aiter_flash" "$capture" &&
    assert_line "RESOLVED_ARG=quantization=aiter_fp4" "$capture"
}

test_atomic_launch_guard() {
  local fake_root="$TMP/fake-project"
  local fake_python="$TMP/fake-python"
  local out_root="$TMP/guarded-output"
  local first_capture="$TMP/guard-first"
  local second_capture="$TMP/guard-second"
  mkdir -p "$fake_root/maxtext"
  cat >"$fake_python" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  exit 0
fi
exit 1
EOF
  chmod +x "$fake_python"

  # The fake executable accepts provenance input, then fails at the would-be
  # MaxText launch without importing JAX.
  env -i PATH="$PATH" HOME="$TMP" PROJECT_ROOT="$fake_root" \
    JAX_AITER_ROOT="$ROOT" MAXTEXT_ROOT="$ROOT" PYTHON_BIN="$fake_python" \
    bash "$PERF_RUNNER" mxfp4 "$out_root" 50 >"$first_capture" 2>&1
  local first_rc="$?"
  [[ "$first_rc" == 1 ]] || {
    printf 'first guarded launch exited %s instead of 1\n' "$first_rc" >&2
    cat "$first_capture" >&2
    return 1
  }
  if env -i PATH="$PATH" HOME="$TMP" PROJECT_ROOT="$fake_root" \
      JAX_AITER_ROOT="$ROOT" MAXTEXT_ROOT="$ROOT" PYTHON_BIN="$fake_python" \
      bash "$PERF_RUNNER" mxfp4 "$out_root" 50 >"$second_capture" 2>&1; then
    printf 'duplicate guarded launch unexpectedly succeeded\n' >&2
    return 1
  fi
  assert_text "train.log already exists; use a fresh output root" "$second_capture" || return 1
  [[ "$(count_prefix "=== RESOLVED_RECIPE_BEGIN ===" "$out_root/mxfp4/train.log")" == 1 ]] || return 1

  # The marker remains authoritative even if a user removes the log.
  rm "$out_root/mxfp4/train.log"
  if env -i PATH="$PATH" HOME="$TMP" PROJECT_ROOT="$fake_root" \
      JAX_AITER_ROOT="$ROOT" MAXTEXT_ROOT="$ROOT" PYTHON_BIN="$fake_python" \
      bash "$PERF_RUNNER" mxfp4 "$out_root" 50 >"$second_capture" 2>&1; then
    printf 'launch with an existing marker unexpectedly succeeded\n' >&2
    return 1
  fi
  assert_text "launch already attempted" "$second_capture"
}

test_log_validator_requires_complete_finite_tail() {
  local log="$TMP/complete-train.log"
  local summary="$TMP/complete-summary.json"
  local step
  : >"$log"
  for ((step = 0; step < 50; ++step)); do
    printf 'completed step: %d, seconds: 1.0, TFLOP/s/device: %d, Tokens/s/device: 200, loss: 1, lm_loss: 1, perplexity: 2, raw_grad_norm: 3, grad_norm: 1, param_norm: 4, lr: 0.001\n' \
      "$step" "$((100 + step))" >>"$log"
  done
  python3 "$LOG_VALIDATOR" "$log" --expected-last-step 49 --tail 10 \
    --out "$summary" >/dev/null &&
    assert_text '"tail_mean_tflops_per_device": 144.5' "$summary" || return 1

  printf 'completed step: 0, seconds: 1.0, TFLOP/s/device: nan\n' \
    >"$TMP/incomplete-train.log"
  ! python3 "$LOG_VALIDATOR" "$TMP/incomplete-train.log" \
    --expected-last-step 49 >/dev/null 2>&1
}

tests=(
  test_perf_mxfp4_direct_mha
  test_perf_plain_fp8
  test_perf_bf16_no_quantization
  test_convergence_modes
  test_rejects_stale_aliases
  test_rejects_noncanonical_mem_fraction
  test_llama31_model_controls_are_always_explicit
  test_blog_profile_rejects_recipe_drift
  test_ci_profile_accepts_explicit_regression_point
  test_installed_wheel_runtime_resolution
  test_atomic_launch_guard
  test_log_validator_requires_complete_finite_tail
)
failures=0
for test_name in "${tests[@]}"; do
  if "$test_name"; then
    printf 'PASS %s\n' "$test_name"
  else
    printf 'FAIL %s\n' "$test_name"
    failures=$((failures + 1))
  fi
done
printf '%d passed, %d failed\n' "$(( ${#tests[@]} - failures ))" "$failures"
exit "$failures"
