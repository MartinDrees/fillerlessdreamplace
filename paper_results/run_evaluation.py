#!/usr/bin/env python3
"""
Run evaluate_ispd.py for fft, mg, mgf, and our placer in parallel and print the results.

Usage:
    python3 run_evaluation.py --bookshelf DIR
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE          = Path(__file__).resolve().parent
EVALUATE       = _HERE / "evaluate_ispd.py"
DREAMPLACE_DIR = str(_HERE / "dreamplace")
ROW_WINDOW     = "10000"


def run_evaluate(args, label):
    cmd = ["python3", str(EVALUATE)] + args
    print(f"[{label}] Starting: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = result.stdout.decode()
    stderr = result.stderr.decode()
    print(f"[{label}] Done (rc={result.returncode})", flush=True)
    if result.returncode != 0:
        print(f"[{label}] STDERR:\n{stderr}", file=sys.stderr)
    return stdout


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run evaluate_ispd.py for all methods in parallel.")
    parser.add_argument("--bookshelf", required=True, metavar="DIR",
                        help="Root directory of the ISPD 2005 bookshelf benchmarks.")
    parsed = parser.parse_args()
    bs = ["--bookshelf", parsed.bookshelf]

    tasks = {
        "fft": ["--dreamplace", DREAMPLACE_DIR, "--method", "fft", "--gp", "--row-window", ROW_WINDOW] + bs,
        "mg":  ["--dreamplace", DREAMPLACE_DIR, "--method", "mg",  "--gp", "--row-window", ROW_WINDOW] + bs,
        "mgf": ["--dreamplace", DREAMPLACE_DIR, "--method", "mgf", "--gp", "--row-window", ROW_WINDOW] + bs,
        "our": ["--our",                                            "--row-window", ROW_WINDOW]             + bs,
    }

    outputs = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_evaluate, task_args, label): label
                   for label, task_args in tasks.items()}
        for fut in as_completed(futures):
            label = futures[fut]
            outputs[label] = fut.result()

    for label in ("fft", "mg", "mgf", "our"):
        print(f"\n{'='*60}")
        print(f"  {label.upper()}")
        print('='*60)
        print(outputs[label])


if __name__ == "__main__":
    main()
