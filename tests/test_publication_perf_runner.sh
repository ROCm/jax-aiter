#!/usr/bin/env bash
# CPU-only resolution tests; no Python, JAX, Docker, or GPU process is started.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PERF_RUNNER="$ROOT/scripts/recipes/run_nvfp4_match_8b.sh"
TTC_RUNNER="$ROOT/scripts/recipes/run_mlperf_ttc_8b.sh"
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

check_common() {
  local capture="$1"
  assert_text "autotune=5 weight_dtype=float32 mu_dtype=float32 use_iota_embed=False batch=4 global_batch=32 sequence=8192 init_seed=0 mem_fraction=.97" "$capture" &&
    assert_line "RESOLVED_ARG=weight_dtype=float32" "$capture" &&
    assert_line "RESOLVED_ARG=mu_dtype=float32" "$capture" &&
    assert_line "RESOLVED_ARG=use_iota_embed=False" "$capture" &&
    assert_line "RESOLVED_ARG=global_batch_size_to_train_on=32" "$capture" &&
    assert_line "RESOLVED_ARG=init_weights_seed=0" "$capture" &&
    assert_line "RESOLVED_ARG=scan_layers=False" "$capture" &&
    assert_line "RESOLVED_ARG=enable_nnx=False" "$capture" &&
    assert_line "RESOLVED_ARG=pure_nnx=False" "$capture" &&
    assert_line "RESOLVED_ARG=pure_nnx_decoder=False" "$capture" &&
    assert_text "project_root=/opt/mxfp4-repro jax_aiter_root=/opt/mxfp4-repro/jax-aiter maxtext_root=/opt/mxfp4-repro/maxtext" "$capture" &&
    assert_text "rocm_safeguards=jax_platforms:rocm,queue_interposition:0,register_enabled:0,no_scratch_reclaim:1,dev_kernarg:1,fine_grain_pcie:1" "$capture" &&
    assert_text "--xla_gpu_autotune_level=5" "$capture" &&
    assert_one_quantization_arg "$capture"
}

test_perf_mxfp4_direct_mha() {
  local capture="$TMP/perf-mxfp4"
  run_perf mxfp4 "$capture" &&
    check_common "$capture" &&
    assert_text "mode=mxfp4 label=MXFP4 processes=1 steps=50 measurement_window=completed_steps_40_49" "$capture" &&
    assert_line "RESOLVED_ARG=quantization=aiter_fp4" "$capture" &&
    assert_line "RESOLVED_ARG=attention=aiter_flash" "$capture" &&
    assert_line "RESOLVED_ARG=use_jax_aiter=True" "$capture" &&
    assert_line "RESOLVED_ARG=remat_policy=minimal_flash_save_fp4col" "$capture" &&
    assert_text "hadamard_passes=wgrad sr_passes=wgrad_col dgrad_partition=gather_packed dgrad_reuse_fwd_col=1 remat_save_col=both pack_gateup_ag=1" "$capture" &&
    assert_text "fp4_select=dispatch fp4_attention_gemm=1 sr_key_mode=maxtext_runtime_params_rng mha_fuse_gqa_reduce=1 mha_zero_pad=1" "$capture" &&
    assert_text "ja_mha_atomic_fp32=0 ja_mha_bf16_cvt=2" "$capture" &&
    assert_absent "RESOLVED_ARG=aiter_attention=False" "$capture"
}

test_perf_te_current_scaling() {
  local capture="$TMP/perf-te"
  run_perf te_fp8_currentscaling "$capture" &&
    check_common "$capture" &&
    assert_text "mode=te_fp8_currentscaling label=TE FP8 current scaling processes=1 steps=50" "$capture" &&
    assert_line "RESOLVED_ARG=quantization=te_fp8_currentscaling" "$capture" &&
    assert_line "RESOLVED_ARG=attention=cudnn_flash_te" "$capture" &&
    assert_line "RESOLVED_ARG=use_jax_aiter=False" "$capture" &&
    assert_text "te_atomic_fp32=1 te_bf16_cvt=2" "$capture" &&
    assert_absent "RESOLVED_ARG=quantization=fp8" "$capture"
}

test_perf_bf16_no_quantization() {
  local capture="$TMP/perf-bf16"
  run_perf bf16 "$capture" &&
    check_common "$capture" &&
    assert_text "mode=bf16 label=BF16 processes=1 steps=50" "$capture" &&
    assert_line "RESOLVED_ARG=quantization=" "$capture" &&
    assert_line "RESOLVED_ARG=attention=cudnn_flash_te" "$capture" &&
    assert_text "quantization=none attention=cudnn_flash_te" "$capture"
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
      check_common "$capture" &&
      assert_text "runner=convergence tag=${mode}_ttc mode=${mode} label=${label} processes=1 steps=6000" "$capture" &&
      assert_line "RESOLVED_ARG=quantization=$quant" "$capture" &&
      assert_line "RESOLVED_ARG=attention=$attention" "$capture" &&
      assert_line "RESOLVED_ARG=enable_mlperf_logging=True" "$capture" || return 1
  done
}

test_rejects_stale_aliases() {
  local alias capture
  for alias in fp8 te_fp8 mxfp4_mlponly; do
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
  assert_text "canonical mem fraction is .97" "$capture"
}

test_atomic_launch_guard() {
  local fake_root="$TMP/fake-project"
  local out_root="$TMP/guarded-output"
  local first_capture="$TMP/guard-first"
  local second_capture="$TMP/guard-second"
  mkdir -p "$fake_root/maxtext"

  # /bin/false stands in for Python: the first launch claims the marker and
  # exits without importing JAX; the second must fail before starting it.
  env -i PATH="$PATH" HOME="$TMP" PROJECT_ROOT="$fake_root" PYTHON_BIN=/bin/false \
    bash "$PERF_RUNNER" mxfp4 "$out_root" 50 >"$first_capture" 2>&1
  [[ "$?" == 1 ]] || {
    printf 'first guarded launch did not reach the fake executable\n' >&2
    return 1
  }
  if env -i PATH="$PATH" HOME="$TMP" PROJECT_ROOT="$fake_root" PYTHON_BIN=/bin/false \
      bash "$PERF_RUNNER" mxfp4 "$out_root" 50 >"$second_capture" 2>&1; then
    printf 'duplicate guarded launch unexpectedly succeeded\n' >&2
    return 1
  fi
  assert_text "train.log already exists; use a fresh output root" "$second_capture" || return 1
  [[ "$(count_prefix "=== RESOLVED_RECIPE_BEGIN ===" "$out_root/mxfp4/train.log")" == 1 ]] || return 1

  # The marker remains authoritative even if a user removes the log.
  rm "$out_root/mxfp4/train.log"
  if env -i PATH="$PATH" HOME="$TMP" PROJECT_ROOT="$fake_root" PYTHON_BIN=/bin/false \
      bash "$PERF_RUNNER" mxfp4 "$out_root" 50 >"$second_capture" 2>&1; then
    printf 'launch with an existing marker unexpectedly succeeded\n' >&2
    return 1
  fi
  assert_text "launch already attempted" "$second_capture"
}

tests=(
  test_perf_mxfp4_direct_mha
  test_perf_te_current_scaling
  test_perf_bf16_no_quantization
  test_convergence_modes
  test_rejects_stale_aliases
  test_rejects_noncanonical_mem_fraction
  test_atomic_launch_guard
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
