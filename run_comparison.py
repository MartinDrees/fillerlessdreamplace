#!/usr/bin/env python3
"""
run_comparison.py
Run DREAMPlace on all ISPD 2005 instances in three modes:
  - fft    : standard FFT-based potential with filler cells
  - mg     : fillerless multigrid potential (25 Jacobi iters/level)
  - mgf    : multigrid potential with filler cells
and print a summary comparison table.

Usage (from DREAMPlace root, inside the container):
  python run_comparison.py [--dry-run]

Results are also written to results/comparison/ispd2005.csv
"""

import subprocess
import sys
import os
import re
import time
import csv
import json
import tempfile
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

PLACER   = "install/dreamplace/Placer.py"
TEST_DIR = "test/ispd2005"
LOG_DIR  = "results/comparison/logs"

RESULT_DIRS = {
    "fft": "results/comparison/fft",
    "mg":  "results/comparison/mg",
    "mgf": "results/comparison/mgf",
}

INSTANCES = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
             "bigblue1",  "bigblue2",  "bigblue3",  "bigblue4"]

MODES = {
    "fft": lambda inst: f"{TEST_DIR}/{inst}.json",
    "mg":  lambda inst: f"{TEST_DIR}/{inst}_multigrid.json",
    "mgf": lambda inst: f"{TEST_DIR}/{inst}_multigrid_fillers.json",
}

# ── log parsing ──────────────────────────────────────────────────────────────

def parse_log(log: str) -> dict:
    """Extract key metrics from a DREAMPlace log string."""
    result = {}

    # GP HPWL: last "wHPWL" value before legalization starts
    wl_matches = re.findall(r"wHPWL\s+([\d.E+\-]+),\s+Overflow\s+([\d.E+\-]+)", log)
    if wl_matches:
        result["gp_hpwl"]    = float(wl_matches[-1][0])
        result["gp_overflow"] = float(wl_matches[-1][1])
        result["gp_iters"]    = len(wl_matches)

    # GP runtime: sum of all "optimizer nesterov takes X.XX seconds" lines
    # (one per global placement stage; typically just one)
    nesterov_times = re.findall(r"optimizer nesterov takes\s+([\d.]+)\s+seconds", log)
    if nesterov_times:
        result["gp_time"] = sum(float(t) for t in nesterov_times)

    # Total runtime: "DREAMPlace - placement takes X.XX seconds"
    m = re.search(r"DREAMPlace - placement takes\s+([\d.]+)\s+seconds", log)
    if m:
        result["total_time"] = float(m.group(1))

    # Post-legalization HPWL: last "wHPWL" in the log
    all_wl = re.findall(r"wHPWL\s+([\d.E+\-]+)", log)
    if all_wl:
        result["legal_hpwl"] = float(all_wl[-1])

    # Number of fillers
    m = re.search(r"#fillers\s*=\s*(\d+)", log)
    if m:
        result["num_fillers"] = int(m.group(1))

    return result


# ── runner ───────────────────────────────────────────────────────────────────

def run_one(json_path: str, log_path: str, result_dir: str) -> dict:
    """Run Placer.py and return parsed metrics, or {} on failure."""
    if DRY_RUN:
        print(f"  [dry-run] would run: python {PLACER} {json_path}  (result_dir={result_dir})")
        return {"gp_hpwl": 0, "gp_overflow": 0, "gp_iters": 0,
                "gp_time": 0, "legal_hpwl": 0, "total_time": 0, "num_fillers": 0}

    # Write temp JSON with result_dir overridden
    with open(json_path) as _f:
        _cfg = json.load(_f)
    _cfg["result_dir"] = result_dir
    _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(_cfg, _tmp, indent=4)
    _tmp.close()

    cmd = ["python", PLACER, _tmp.name]
    print(f"  running: {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        log = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return {}
    wall = time.time() - t0
    os.unlink(_tmp.name)

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(log)

    if proc.returncode != 0:
        print(f"  FAILED (rc={proc.returncode}), log -> {log_path}")
        return {}

    metrics = parse_log(log)
    metrics["wall_time"] = wall
    gp_t = metrics.get("gp_time", float("nan"))
    gp_t_str = f"{gp_t:.1f}s" if gp_t == gp_t else "?"
    print(f"  done in {wall:.1f}s (GP {gp_t_str}), "
          f"GP HPWL={metrics.get('gp_hpwl','?'):.4e}, "
          f"legal HPWL={metrics.get('legal_hpwl','?'):.4e}")
    return metrics


# ── formatting helpers ───────────────────────────────────────────────────────

def fmt(v, spec=".4e"):
    return format(v, spec) if v == v else "N/A"

def diff_pct(baseline, other):
    """(other - baseline) / baseline * 100, or nan."""
    if baseline and other and baseline == baseline and other == other:
        return 100.0 * (other - baseline) / baseline
    return float("nan")

def speedup(base_t, other_t):
    """base_t / other_t, or nan."""
    if base_t and other_t and base_t == base_t and other_t == other_t:
        return base_t / other_t
    return float("nan")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # { instance -> { mode -> metrics } }
    data = {inst: {} for inst in INSTANCES}

    for inst in INSTANCES:
        for mode, json_fn in MODES.items():
            json_path = json_fn(inst)
            log_path  = f"{LOG_DIR}/{inst}_{mode}.log"
            print(f"\n{'='*60}")
            print(f"  {inst}  [{mode}]")
            print(f"{'='*60}")
            data[inst][mode] = run_one(json_path, log_path, RESULT_DIRS[mode])

    # ── table ────────────────────────────────────────────────────────────────
    #
    # Columns:
    #   Instance | FFT fillers
    #   | FFT GP-HPWL | FFT leg-HPWL | FFT GP-t | FFT tot-t
    #   | MG  GP-HPWL | MG  leg-HPWL | MG  GP-t | MG  tot-t | MG leg-diff% | MG GP-spdup
    #   | MGF GP-HPWL | MGF leg-HPWL | MGF GP-t | MGF tot-t | MGF leg-diff% | MGF GP-spdup

    COL = {
        "inst":          12,
        "fillers":       12,
        "gp_hpwl":       13,
        "leg_hpwl":      13,
        "gp_t":          10,
        "tot_t":         11,
        "diff":           9,
        "spdup":          8,
    }

    def hdr(s, w): return f"{s:>{w}}"

    header = (
        f"{'Instance':<{COL['inst']}} "
        + hdr("FFT fillers",   COL["fillers"])  + " "
        + hdr("FFT GP-HPWL",  COL["gp_hpwl"])  + " "
        + hdr("FFT leg-HPWL", COL["leg_hpwl"]) + " "
        + hdr("FFT GP-t",     COL["gp_t"])      + " "
        + hdr("FFT tot-t",    COL["tot_t"])      + " "
        + hdr("MG GP-HPWL",   COL["gp_hpwl"])  + " "
        + hdr("MG leg-HPWL",  COL["leg_hpwl"]) + " "
        + hdr("MG GP-t",      COL["gp_t"])      + " "
        + hdr("MG tot-t",     COL["tot_t"])      + " "
        + hdr("MG diff%",     COL["diff"])       + " "
        + hdr("MG spdup",     COL["spdup"])      + " "
        + hdr("MGF GP-HPWL",  COL["gp_hpwl"])  + " "
        + hdr("MGF leg-HPWL", COL["leg_hpwl"]) + " "
        + hdr("MGF GP-t",     COL["gp_t"])      + " "
        + hdr("MGF tot-t",    COL["tot_t"])      + " "
        + hdr("MGF diff%",    COL["diff"])       + " "
        + hdr("MGF spdup",    COL["spdup"])
    )
    sep = "-" * len(header)
    print(f"\n\n{sep}")
    print(header)
    print(sep)

    csv_rows = []
    for inst in INSTANCES:
        fft = data[inst].get("fft", {})
        mg  = data[inst].get("mg",  {})
        mgf = data[inst].get("mgf", {})

        # FFT baseline values
        gp_hpwl_fft  = fft.get("gp_hpwl",    float("nan"))
        leg_hpwl_fft = fft.get("legal_hpwl",  float("nan"))
        gp_t_fft     = fft.get("gp_time",     float("nan"))
        tot_t_fft    = fft.get("total_time",   float("nan"))
        fillers      = fft.get("num_fillers",  0)

        # MG (fillerless multigrid)
        gp_hpwl_mg   = mg.get("gp_hpwl",    float("nan"))
        leg_hpwl_mg  = mg.get("legal_hpwl",  float("nan"))
        gp_t_mg      = mg.get("gp_time",     float("nan"))
        tot_t_mg     = mg.get("total_time",   float("nan"))

        # MGF (multigrid + fillers)
        gp_hpwl_mgf  = mgf.get("gp_hpwl",   float("nan"))
        leg_hpwl_mgf = mgf.get("legal_hpwl", float("nan"))
        gp_t_mgf     = mgf.get("gp_time",    float("nan"))
        tot_t_mgf    = mgf.get("total_time",  float("nan"))

        # Derived: diff% and speedup vs FFT baseline
        mg_diff   = diff_pct(leg_hpwl_fft, leg_hpwl_mg)
        mg_spdup  = speedup(gp_t_fft, gp_t_mg)
        mgf_diff  = diff_pct(leg_hpwl_fft, leg_hpwl_mgf)
        mgf_spdup = speedup(gp_t_fft, gp_t_mgf)

        w = COL
        print(
            f"{inst:<{w['inst']}} "
            + f"{fillers:>{w['fillers']},} "
            + f"{fmt(gp_hpwl_fft):>{w['gp_hpwl']}} "
            + f"{fmt(leg_hpwl_fft):>{w['leg_hpwl']}} "
            + f"{fmt(gp_t_fft, '.1f'):>{w['gp_t']}} "
            + f"{fmt(tot_t_fft, '.1f'):>{w['tot_t']}} "
            + f"{fmt(gp_hpwl_mg):>{w['gp_hpwl']}} "
            + f"{fmt(leg_hpwl_mg):>{w['leg_hpwl']}} "
            + f"{fmt(gp_t_mg, '.1f'):>{w['gp_t']}} "
            + f"{fmt(tot_t_mg, '.1f'):>{w['tot_t']}} "
            + f"{fmt(mg_diff, '+.2f'):>{w['diff']}} "
            + f"{fmt(mg_spdup, '.2f'):>{w['spdup']}} "
            + f"{fmt(gp_hpwl_mgf):>{w['gp_hpwl']}} "
            + f"{fmt(leg_hpwl_mgf):>{w['leg_hpwl']}} "
            + f"{fmt(gp_t_mgf, '.1f'):>{w['gp_t']}} "
            + f"{fmt(tot_t_mgf, '.1f'):>{w['tot_t']}} "
            + f"{fmt(mgf_diff, '+.2f'):>{w['diff']}} "
            + f"{fmt(mgf_spdup, '.2f'):>{w['spdup']}}"
        )

        csv_rows.append({
            "instance":           inst,
            "fft_num_fillers":    fillers,
            "fft_gp_hpwl":        gp_hpwl_fft,
            "fft_legal_hpwl":     leg_hpwl_fft,
            "fft_gp_time_s":      gp_t_fft,
            "fft_total_time_s":   tot_t_fft,
            "mg_gp_hpwl":         gp_hpwl_mg,
            "mg_legal_hpwl":      leg_hpwl_mg,
            "mg_gp_time_s":       gp_t_mg,
            "mg_total_time_s":    tot_t_mg,
            "mg_legal_hpwl_diff_pct": mg_diff,
            "mg_gp_speedup":      mg_spdup,
            "mgf_gp_hpwl":        gp_hpwl_mgf,
            "mgf_legal_hpwl":     leg_hpwl_mgf,
            "mgf_gp_time_s":      gp_t_mgf,
            "mgf_total_time_s":   tot_t_mgf,
            "mgf_legal_hpwl_diff_pct": mgf_diff,
            "mgf_gp_speedup":     mgf_spdup,
        })

    print(sep)
    print("diff%  = (mode - FFT) / FFT * 100 on leg-HPWL  (negative = better than FFT)")
    print("spdup  = FFT_GP_time / mode_GP_time")
    print("MG     = multigrid fillerless   MGF = multigrid + fillers")

    # ── write CSV ─────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    csv_path = "results/comparison/ispd2005.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nCSV written to {csv_path}")


if __name__ == "__main__":
    main()
