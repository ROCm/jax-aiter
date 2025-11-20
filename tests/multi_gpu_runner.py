#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import threading

# Default configuration
DEFAULT_NGPUS = 8
DEFAULT_TESTPATH = "tests"  # Changed from single file to entire tests directory
DEFAULT_RERUNS = "3"


def detect_gpu_count():
    """Try to detect the number of available AMD GPUs."""
    # Try amd-smi first (newer tool)
    try:
        result = subprocess.run(
            ["amd-smi", "list", "--json"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Count occurrences of "gpu" in the JSON output
            gpu_count = result.stdout.count('"gpu"')
            if gpu_count > 0:
                print(f"Detected {gpu_count} GPUs using amd-smi")
                return gpu_count
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fall back to rocm-smi
    try:
        result = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Count lines that start with a digit (GPU IDs)
            gpu_count = len(
                [
                    line
                    for line in result.stdout.splitlines()
                    if line and line[0].isdigit()
                ]
            )
            if gpu_count > 0:
                print(f"Detected {gpu_count} GPUs using rocm-smi")
                return gpu_count
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check environment variables
    if "HIP_VISIBLE_DEVICES" in os.environ:
        devices = os.environ["HIP_VISIBLE_DEVICES"].split(",")
        print(f"Using HIP_VISIBLE_DEVICES: {len(devices)} GPUs")
        return len(devices)

    # Default fallback
    print(f"Could not detect GPUs, defaulting to {DEFAULT_NGPUS}")
    return DEFAULT_NGPUS


def collect_all_tests(testpath):
    """Return list of nodeids collected by pytest."""
    out = subprocess.check_output(
        ["pytest", "--collect-only", "-q", testpath],
        text=True,
    )
    # pytest --collect-only -q prints one nodeid per line.
    nodeids = []
    for line in out.splitlines():
        line = line.strip()
        if line and not line.startswith("<") and "::" in line:
            nodeids.append(line)
    return nodeids


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
    shard_total = how many tests this shard should run.
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

        # Only count explicit status markers to avoid false positives from output
        # that contains dots (like ".Output max diff: 0.015625").
        counted_this_line = False

        if " PASSED" in line:
            done += 1
            passed += 1
            counted_this_line = True
        elif " FAILED" in line:
            done += 1
            failed += 1
            counted_this_line = True
            # Extract test nodeid from the line.
            # Pytest verbose format: "test_file.py::test_name FAILED [ 24%]".
            # Match any .py file path followed by :: and test name.
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

        # Count pytest progress indicators from the start of the line.
        if not counted_this_line:
            if stripped:
                pytest_chars = 0
                for char in stripped:
                    if char in ".FsxR":  # Added R for reruns
                        pytest_chars += 1
                    else:
                        break
                if pytest_chars > 0:
                    status_prefix = stripped[:pytest_chars]
                    dots = status_prefix.count(".")
                    fails = status_prefix.count("F")
                    skips = status_prefix.count("s")
                    rerun_markers = status_prefix.count("R")
                    # R doesn't increment 'done' as it's a retry, not a new test
                    done += dots + fails + skips
                    passed += dots
                    failed += fails
                    skipped += skips
                    reruns += rerun_markers

        prefix = f"[GPU {gpu_idx} {done}/{shard_total}]"
        if stripped:
            sys.stdout.write(f"{prefix} {line}")
        else:
            sys.stdout.write(line)
        sys.stdout.flush()

    with stats_lock:
        gpu_stats[gpu_idx] = {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "reruns": reruns,
            "total": done,
            "failed_tests": failed_tests,
        }


def write_failed_tests(output_file):
    """Write all failed test nodeids to a file."""
    all_failed = []

    for gpu_idx in sorted(gpu_stats.keys()):
        stats = gpu_stats[gpu_idx]
        gpu_failed = stats.get("failed_tests", [])
        if gpu_failed:
            print(f"  GPU {gpu_idx}: {len(gpu_failed)} failed tests")
        all_failed.extend(gpu_failed)

    print(f"\nTotal failed tests collected: {len(all_failed)}")

    if all_failed:
        with open(output_file, "w") as f:
            for test in all_failed:
                f.write(f"{test}\n")
        print(f"Failed tests saved to: {output_file}")
    else:
        print("No failed tests to save.")
        print(
            "Note: If you expected failed tests, check that tests are actually failing."
        )


def print_final_report(exitcodes, total_expected):
    """Print a nice summary report of all test results."""
    print(f"\n{'='*80}")
    print("FINAL TEST REPORT")
    print(f"{'='*80}")
    print(f"Total tests expected: {total_expected}\n")

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    total_reruns = 0
    total_tests = 0

    for gpu_idx in sorted(gpu_stats.keys()):
        stats = gpu_stats[gpu_idx]
        exitcode = exitcodes[gpu_idx] if gpu_idx < len(exitcodes) else -1

        print(
            f"GPU {gpu_idx}: "
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
        print(f"\nOVERALL: FAILED")
    elif total_passed > 0:
        print(f"\nOVERALL: PASSED")
    else:
        print(f"\nOVERALL: NO TESTS RUN")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run tests in parallel across multiple AMD GPUs"
    )
    parser.add_argument(
        "testpath",
        nargs="?",
        default=DEFAULT_TESTPATH,
        help=f"Path to test file or directory (default: {DEFAULT_TESTPATH})",
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        default=None,
        help=f"Number of GPUs to use (default: auto-detect or {DEFAULT_NGPUS})",
    )
    parser.add_argument(
        "--reruns",
        type=int,
        default=DEFAULT_RERUNS,
        help=f"Number of reruns for failed tests (default: {DEFAULT_RERUNS})",
    )
    parser.add_argument(
        "--save-failed",
        metavar="FILE",
        help="Save failed test nodeids to the specified file",
    )
    parser.add_argument("--filter", help="Filter tests using pytest -k expression")
    parser.add_argument(
        "--maxfail", type=int, help="Exit after first N failures per GPU"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose pytest output"
    )
    args = parser.parse_args()

    # Determine number of GPUs
    if args.ngpus is not None:
        ngpus = args.ngpus
        print(f"Using specified {ngpus} GPUs")
    else:
        ngpus = detect_gpu_count()

    # Collect tests
    print(f"Collecting tests from {args.testpath}...")
    all_nodeids = collect_all_tests(args.testpath)

    if not all_nodeids:
        print("No tests found!")
        sys.exit(1)

    shards = make_shards(all_nodeids, ngpus)

    total_tests = len(all_nodeids)
    print(f"\n{'='*80}")
    print(f"Total tests collected: {total_tests}")
    print(f"Distributing across {ngpus} GPUs (~{total_tests//ngpus} tests per GPU)")
    print(f"Test path: {args.testpath}")
    if args.filter:
        print(f"Filter: {args.filter}")
    print(f"{'='*80}\n")

    procs = []
    threads = []

    for gpu in range(ngpus):
        shard_tests = shards[gpu]
        shard_total = len(shard_tests)

        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(gpu)
        env["PYTEST_SHARD_TOTAL"] = str(ngpus)
        env["PYTEST_SHARD_INDEX"] = str(gpu)

        cmd = [
            "pytest",
            "--no-header",
            "--no-summary",
            "--disable-warnings",
            "--color=no",
            "-q",
            f"--reruns={args.reruns}",
            args.testpath,
        ]

        # Note: -q and -v are mutually exclusive, so we skip -v when using -q
        # if args.verbose:
        #     cmd.append("-v")

        if args.maxfail:
            cmd.append(f"--maxfail={args.maxfail}")

        if args.filter:
            cmd.extend(["-k", args.filter])

        print(
            f"[launcher] Starting shard {gpu}/{ngpus-1} on GPU {gpu} with {shard_total} tests"
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

    print()  # blank line before test output starts.

    try:
        exitcodes = [p.wait() for p in procs]
    except KeyboardInterrupt:
        print("\n[launcher] Ctrl-C, terminating...")
        for p in procs:
            p.terminate()
        exitcodes = [p.wait() for p in procs]

    for t in threads:
        t.join(timeout=1.0)

    print_final_report(exitcodes, total_tests)

    if args.save_failed:
        write_failed_tests(args.save_failed)

    bad = [c for c in exitcodes if c]
    sys.exit(bad[0] if bad else 0)


if __name__ == "__main__":
    main()
