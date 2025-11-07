# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
"""
Smoke test runner for JAX-aiter MHA implementation.
Runs a representative subset of tests for quick validation.
"""

import argparse
import os
import re
import subprocess
import sys
import threading
from typing import List, Dict, Set


def get_available_gpus():
    """Get number of available GPUs from HIP_VISIBLE_DEVICES."""
    hip_devices = os.environ.get("HIP_VISIBLE_DEVICES", "")
    if hip_devices:
        gpu_list = [g.strip() for g in hip_devices.split(",") if g.strip()]
        return len(gpu_list)
    else:
        # If not set, assume all 4 GPUs are available.
        return 4


NGPUS = get_available_gpus()
TESTFILE = "tests/test_mha_ja.py"
RERUNS = "2"  # Fewer reruns for smoke tests

# Define smoke test filters for different focus areas.
SMOKE_FILTERS = {
    "basic": {
        "dimensions": ["32", "128"],
        "seqlens": ["128", "512"],
        "mha_types": ["mha"],
        "dtypes": ["bfloat16"],
        "features": ["False-False-no"],  # causal-local-bias_type
        "layouts": ["BSHD"],
        "padding": ["no_padding"],
    },
    "padding": {
        "dimensions": ["32", "128"],
        "seqlens": ["256", "512"],
        "mha_types": ["mha", "gqa"],
        "dtypes": ["bfloat16", "float16"],
        "features": ["False-False-no", "True-False-no"],  # Add causal
        "layouts": ["BSHD"],
        "padding": ["mixed", "q_only", "k_only", "no_padding"],
    },
    "mha_types": {
        "dimensions": ["64", "128"],
        "seqlens": ["256", "512"],
        "mha_types": ["mha", "mqa", "gqa"],
        "dtypes": ["bfloat16"],
        "features": ["False-False-no", "True-False-no"],
        "layouts": ["BSHD", "KVPACKED"],
        "padding": ["no_padding"],
    },
    "features": {
        "dimensions": ["32", "128"],
        "seqlens": ["256", "512"],
        "mha_types": ["mha"],
        "dtypes": ["bfloat16", "float16"],
        "features": [
            "False-False-no",  # baseline
            "True-False-no",  # causal
            "False-True-no",  # local
            "False-False-bias",  # bias
            "True-False-alibi",  # alibi + causal
        ],
        "layouts": ["BSHD"],
        "padding": ["no_padding", "mixed"],
    },
    "comprehensive": {
        "dimensions": ["32", "64", "128"],
        "seqlens": ["128", "256", "512"],
        "mha_types": ["mha", "mqa", "gqa"],
        "dtypes": ["bfloat16", "float16"],
        "features": [
            "False-False-no",
            "True-False-no",
            "False-True-no",
            "False-False-bias",
            "True-False-alibi",
        ],
        "layouts": ["BSHD", "KVPACKED"],
        "padding": ["mixed", "no_padding"],
    },
}


def collect_all_tests():
    """Return list of nodeids collected by pytest."""
    out = subprocess.check_output(
        ["pytest", "--collect-only", "-q", TESTFILE],
        text=True,
    )
    # pytest --collect-only -q prints one nodeid per line.
    return [
        line.strip()
        for line in out.splitlines()
        if line.strip() and not line.startswith("<")
    ]


def filter_smoke_tests(all_tests: List[str], focus: str) -> List[str]:
    """Filter tests based on smoke test criteria."""
    filters = SMOKE_FILTERS[focus]
    filtered_tests = []

    for test in all_tests:
        # Check if this is a test we want to run.
        should_include = False

        # Parse test parameters from the test node ID.
        # Format: test_mha_ja.py::test_flash_attn_output[params] or test_flash_attn_seq_padding[params].

        # Check if it's a padding test.
        if "test_flash_attn_seq_padding" in test:
            # For padding tests, check padding scenarios.
            for padding_type in filters["padding"]:
                if f"[{padding_type}-" in test or f"-{padding_type}-" in test:
                    # Also check dimensions.
                    for dim in filters["dimensions"]:
                        if f"-{dim}-{dim}-" in test or f"d{dim}-" in test:
                            should_include = True
                            break
                    if should_include:
                        break

        elif "test_flash_attn_output" in test:
            # For main tests, check all criteria.
            matches = {
                "dimension": False,
                "seqlen": False,
                "mha_type": False,
                "dtype": False,
                "feature": False,
                "layout": False,
            }

            # Check dimensions (format: -32-32- or similar).
            for dim in filters["dimensions"]:
                if f"-{dim}-{dim}-" in test or f"-{dim}-" in test:
                    matches["dimension"] = True
                    break

            # Check sequence lengths.
            for seqlen in filters["seqlens"]:
                if f"-{seqlen}-{seqlen}" in test or f"-{seqlen}-" in test:
                    matches["seqlen"] = True
                    break

            # Check MHA types.
            for mha_type in filters["mha_types"]:
                if f"-{mha_type}-" in test:
                    matches["mha_type"] = True
                    break

            # Check dtypes.
            for dtype in filters["dtypes"]:
                if f"-{dtype}]" in test or f"-{dtype}-" in test:
                    matches["dtype"] = True
                    break

            # Check features (bias_type-local-causal).
            for feature in filters["features"]:
                if feature in test:
                    matches["feature"] = True
                    break

            # Check layouts.
            for layout in filters["layouts"]:
                if layout in test:
                    matches["layout"] = True
                    break

            # Include test if it matches criteria from each category.
            if all(matches.values()):
                should_include = True

        if should_include:
            filtered_tests.append(test)

    return filtered_tests


def make_shards(nodeids, n):
    """Return list of lists: shard[i] = tests for shard i."""
    shards = [[] for _ in range(n)]
    for i, nid in enumerate(nodeids):
        shards[i % n].append(nid)
    return shards


# Shared structure to collect stats.
stats_lock = threading.Lock()
gpu_stats = {}


def stream_output(gpu_idx, proc, shard_total):
    """
    Read pytest output and print with a real progress prefix.
    shard_total = how many tests this shard should run
    """
    done = 0
    passed = 0
    failed = 0
    skipped = 0
    reruns = 0
    failed_tests = []

    for raw in proc.stdout:
        line = raw.decode(errors="ignore")
        stripped = line.strip()

        # Only count explicit status markers.
        counted_this_line = False

        if " PASSED" in line:
            done += 1
            passed += 1
            counted_this_line = True
        elif " FAILED" in line:
            done += 1
            failed += 1
            counted_this_line = True
            # Extract test nodeid.
            match = re.search(r"(\S+\.py::\S+)\s+FAILED\s*(?:\[\s*\d+%\s*\])?", line)
            if match:
                test_id = match.group(1)
                if test_id not in failed_tests:
                    failed_tests.append(test_id)
        elif " SKIPPED" in line:
            done += 1
            skipped += 1
            counted_this_line = True
        elif " RERUN" in line:
            reruns += 1

        # Count pytest progress indicators
        if not counted_this_line:
            if stripped:
                pytest_chars = 0
                for char in stripped:
                    if char in ".Fsx":
                        pytest_chars += 1
                    else:
                        break
                if pytest_chars > 0:
                    status_prefix = stripped[:pytest_chars]
                    dots = status_prefix.count(".")
                    fails = status_prefix.count("F")
                    skips = status_prefix.count("s")
                    done += dots + fails + skips
                    passed += dots
                    failed += fails
                    skipped += skips

        # Print with progress prefix
        prefix = f"[GPU {gpu_idx} {done}/{shard_total}]"
        if stripped:
            sys.stdout.write(f"{prefix} {line}")
        else:
            sys.stdout.write(line)
        sys.stdout.flush()

    # Store final stats
    with stats_lock:
        gpu_stats[gpu_idx] = {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "reruns": reruns,
            "total": done,
            "failed_tests": failed_tests,
        }


def print_final_report(exitcodes, total_expected, focus):
    """Print a nice summary report of all test results."""
    print(f"\n{'='*80}")
    print(f"SMOKE TEST REPORT - Focus: {focus.upper()}")
    print(f"{'='*80}")
    print(f"Total smoke tests selected: {total_expected}\n")

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    total_reruns = 0
    total_tests = 0

    for gpu_idx in sorted(gpu_stats.keys()):
        stats = gpu_stats[gpu_idx]
        exitcode = exitcodes[gpu_idx] if gpu_idx < len(exitcodes) else -1
        status_icon = "✓" if exitcode == 0 else "✗"

        print(
            f"GPU {gpu_idx} {status_icon}: "
            f"{stats['passed']} passed, "
            f"{stats['failed']} failed, "
            f"{stats['skipped']} skipped, "
            f"{stats['reruns']} reruns"
        )

        total_passed += stats["passed"]
        total_failed += stats["failed"]
        total_skipped += stats["skipped"]
        total_reruns += stats["reruns"]
        total_tests += stats["total"]

    print(f"\n{'-'*80}")
    print(
        f"TOTAL: {total_passed} passed, {total_failed} failed, "
        f"{total_skipped} skipped, {total_reruns} reruns"
    )
    print(f"Total tests completed: {total_tests}")

    if total_failed > 0:
        print(f"\nSMOKE TEST: FAILED")
        print("\nFailed tests:")
        for gpu_idx in sorted(gpu_stats.keys()):
            for failed_test in gpu_stats[gpu_idx]["failed_tests"]:
                print(f"  - {failed_test}")
    elif total_passed > 0:
        print(f"\nSMOKE TEST: PASSED")
    else:
        print(f"\nSMOKE TEST: NO TESTS RUN")

    print(f"{'='*80}\n")


def estimate_runtime(num_tests, focus):
    """Estimate runtime based on number of tests and focus area."""
    # Rough estimates based on test complexity.
    time_per_test = {
        "basic": 2,  # seconds
        "padding": 3,
        "mha_types": 2.5,
        "features": 3,
        "comprehensive": 3,
    }

    seconds = (num_tests * time_per_test[focus]) / NGPUS
    minutes = seconds / 60

    return minutes


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Run smoke tests for JAX-aiter MHA implementation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--focus",
        choices=["basic", "padding", "mha_types", "features", "comprehensive"],
        default="basic",
        help="Focus area for smoke tests",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list tests that would be run, don't execute",
    )
    parser.add_argument(
        "--save-failed",
        metavar="FILE",
        help="Save failed test nodeids to the specified file",
    )

    args = parser.parse_args()

    # Collect all tests.
    print("Collecting all tests...")
    all_tests = collect_all_tests()
    print(f"Total tests available: {len(all_tests)}")

    # Filter to smoke test subset.
    print(f"\nFiltering for smoke tests (focus: {args.focus})...")
    smoke_tests = filter_smoke_tests(all_tests, args.focus)
    print(f"Smoke tests selected: {len(smoke_tests)}")

    if len(smoke_tests) == 0:
        print("ERROR: No tests matched smoke test criteria!")
        sys.exit(1)

    # Estimate runtime.
    est_minutes = estimate_runtime(len(smoke_tests), args.focus)
    print(f"Estimated runtime: ~{est_minutes:.1f} minutes")

    if args.list_only:
        print("\nTests that would be run:")
        for test in sorted(smoke_tests):
            print(f"  {test}")
        sys.exit(0)

    # Create shards.
    shards = make_shards(smoke_tests, NGPUS)

    print(f"\n{'='*80}")
    print(f"Running {len(smoke_tests)} smoke tests")
    print(
        f"Distributing across {NGPUS} GPUs (~{len(smoke_tests)//NGPUS} tests per GPU)"
    )
    print(f"Focus area: {args.focus}")
    print(f"{'='*80}\n")

    procs = []
    threads = []

    for gpu in range(NGPUS):
        shard_tests = shards[gpu]
        shard_total = len(shard_tests)

        if shard_total == 0:
            print(f"[launcher] GPU {gpu}: No tests to run")
            continue

        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(gpu)

        # Build pytest command with specific test node IDs.
        cmd = [
            "pytest",
            "-v",
            "--no-header",
            "--no-summary",
            "--disable-warnings",
            f"--reruns={RERUNS}",
        ]

        # Add specific test node IDs.
        cmd.extend(shard_tests)

        print(
            f"[launcher] Starting shard {gpu}/{NGPUS-1} on GPU {gpu} with {shard_total} tests"
        )
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append(proc)

        t = threading.Thread(
            target=stream_output,
            args=(gpu, proc, shard_total),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Wait for processes.
    try:
        exitcodes = [p.wait() for p in procs]
    except KeyboardInterrupt:
        print("\n[launcher] Ctrl-C, terminating...")
        for p in procs:
            p.terminate()
        exitcodes = [p.wait() for p in procs]

    for t in threads:
        t.join(timeout=1.0)

    print_final_report(exitcodes, len(smoke_tests), args.focus)

    if args.save_failed and gpu_stats:
        all_failed = []
        for gpu_idx in gpu_stats:
            all_failed.extend(gpu_stats[gpu_idx].get("failed_tests", []))

        if all_failed:
            with open(args.save_failed, "w") as f:
                for test in all_failed:
                    f.write(f"{test}\n")
            print(f"Failed tests saved to: {args.save_failed}")

    bad = [c for c in exitcodes if c]
    sys.exit(bad[0] if bad else 0)


if __name__ == "__main__":
    main()
