#!/usr/bin/env python3
"""
Evaluate chip runs across all ISPD 2005 benchmarks (HPWL + legality).

Usage:
    python3 evaluate_ispd.py --bookshelf DIR --our [--row-window N]
    python3 evaluate_ispd.py --bookshelf DIR --dreamplace [DIR] --method METHOD [--gp] [--row-window N]

Examples:
    python3 evaluate_ispd.py --bookshelf /path/to/bookshelf --our
    python3 evaluate_ispd.py --bookshelf /path/to/bookshelf --dreamplace dreamplace --method fft
    python3 evaluate_ispd.py --bookshelf /path/to/bookshelf --dreamplace dreamplace --method fft --gp

Our placer mode (--our):
  For each chip in {adaptec1..4, bigblue1..4}, reads
    our/<chip>/<chip>_our.dp.pl  - final placement (post DREAMPlace DP)
    our/<chip>/<chip>_our.gp.pl  - our GP output (for GP HPWL)
  Evaluates HPWL and legality. No timing information is reported.

DREAMPlace mode (--dreamplace DIR --method METHOD):
  Reads placements from DIR/<chip>/<chip>_<method>.dp.pl.
  All methods (fft, mg, mgf) are stored under the same DIR/<chip>/ directory.
  DIR defaults to paper_results/dreamplace, METHOD defaults to fft.
  With --gp: also compute HPWL on DIR/<chip>/<chip>_<method>.gp.pl and
  show as a separate column. Legality check is only on the final .dp.pl.

--bookshelf DIR is required. Each chip is expected at DIR/<chip>/<chip>.nodes etc.

--row-window controls the ROW_JUMP parameter of legal2.pl (default: 890).
Larger values process more rows per pass (faster, more memory).
"""

import subprocess
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# -- Paths -------------------------------------------------------------------
_HERE           = Path(__file__).resolve().parent
DREAMPLACE_ROOT = _HERE / "dreamplace"
OUR_ROOT        = _HERE / "our"
HPWL_SCRIPT     = _HERE / "hpwl.pl"
LEGAL_SCRIPT    = _HERE / "legal2.pl"


CHIPS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
         "bigblue1",  "bigblue2",  "bigblue3",  "bigblue4"]

LEGAL_ROW_WINDOW_DEFAULT = 890


def run_cmd(cmd):
    """Run a command, return (stdout, stderr, returncode)."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode(), result.stderr.decode(), result.returncode


def _evaluate_transformed(chip, transformed, bs_dir, row_window, result):
    """Run HPWL and legality checks on an already-transformed .pl file."""
    nodes = bs_dir / ("%s.nodes" % chip)
    pl    = bs_dir / ("%s.pl"    % chip)
    nets  = bs_dir / ("%s.nets"  % chip)
    scl   = bs_dir / ("%s.scl"   % chip)

    # HPWL
    stdout, stderr, rc = run_cmd(
        ["perl", str(HPWL_SCRIPT), str(nodes), str(pl), str(transformed), str(nets)]
    )
    if rc != 0:
        result["error"] = "hpwl.pl failed:\n%s" % stderr.strip()
        return result

    m = re.search(r"Total HPWL:\s*(\d+)", stdout)
    if m:
        result["hpwl"] = int(m.group(1))
    else:
        result["error"] = "Could not parse HPWL from output:\n%s" % stdout.strip()
        return result

    # Legality
    stdout, stderr, rc = run_cmd(
        ["perl", str(LEGAL_SCRIPT), str(nodes), str(pl), str(transformed), str(scl),
         str(row_window)]
    )
    counts = {}
    for line in stdout.splitlines():
        m = re.match(r"^(\d)\s+(\d+)$", line.strip())
        if m:
            counts[int(m.group(1))] = int(m.group(2))

    if counts:
        total_errors = sum(counts.values())
        if total_errors == 0:
            result["legal"] = "LEGAL (0 errors)"
        else:
            parts = ", ".join("Type%d=%d" % (t, n) for t, n in sorted(counts.items()) if n > 0)
            result["legal"] = "ILLEGAL (%s)" % parts
    else:
        result["legal"] = "UNKNOWN (could not parse legality output)"

    return result


def evaluate_chip_our(chip, bookshelf_root, row_window):
    """Evaluate our placer's result for one chip.

    Uses:
      our/<chip>/<chip>_our.dp.pl  - final placement after DREAMPlace DP
      our/<chip>/<chip>_our.gp.pl  - GP placement (for GP HPWL)
    """
    chip_dir = OUR_ROOT / chip
    bs_dir   = bookshelf_root / chip
    result   = {"chip": chip, "hpwl": None, "hpwl_gp": None, "legal": None, "error": None}

    if not chip_dir.is_dir():
        result["error"] = "our dir not found: %s" % chip_dir
        return result

    final_pl = chip_dir / ("%s_our.dp.pl" % chip)
    if not final_pl.exists():
        result["error"] = "%s_our.dp.pl missing" % chip
        return result

    if not bs_dir.is_dir():
        result["error"] = "bookshelf dir not found: %s" % bs_dir
        return result

    # GP HPWL from <chip>_our.gp.pl
    gp_pl = chip_dir / ("%s_our.gp.pl" % chip)
    if gp_pl.exists():
        nodes = bs_dir / ("%s.nodes" % chip)
        pl    = bs_dir / ("%s.pl"    % chip)
        nets  = bs_dir / ("%s.nets"  % chip)
        stdout, stderr, rc = run_cmd(
            ["perl", str(HPWL_SCRIPT), str(nodes), str(pl), str(gp_pl), str(nets)]
        )
        if rc == 0:
            m = re.search(r"Total HPWL:\s*(\d+)", stdout)
            if m:
                result["hpwl_gp"] = int(m.group(1))

    return _evaluate_transformed(chip, final_pl, bs_dir, row_window, result)


def evaluate_chip_dreamplace(chip, result_dir, method, bookshelf_root, row_window, gp=False):
    """Evaluate a DREAMPlace result for one chip.

    Expects: result_dir/<chip>/<chip>_<method>.dp.pl
    """
    bs_dir      = bookshelf_root / chip
    chip_dir    = result_dir / chip
    transformed = chip_dir / ("%s_%s.dp.pl" % (chip, method))
    result      = {"chip": chip, "hpwl": None, "hpwl_gp": None, "legal": None, "error": None}

    if not transformed.exists():
        result["error"] = "placement file not found: %s" % transformed
        return result

    if not bs_dir.is_dir():
        result["error"] = "bookshelf dir not found: %s" % bs_dir
        return result

    if gp:
        raw_pl = chip_dir / ("%s_%s.gp.pl" % (chip, method))
        if raw_pl.exists():
            nodes = bs_dir / ("%s.nodes" % chip)
            pl    = bs_dir / ("%s.pl"    % chip)
            nets  = bs_dir / ("%s.nets"  % chip)
            stdout, stderr, rc = run_cmd(
                ["perl", str(HPWL_SCRIPT), str(nodes), str(pl), str(raw_pl), str(nets)]
            )
            if rc == 0:
                m = re.search(r"Total HPWL:\s*(\d+)", stdout)
                if m:
                    result["hpwl_gp"] = int(m.group(1))

    return _evaluate_transformed(chip, transformed, bs_dir, row_window, result)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate ISPD 2005 chip runs (HPWL + legality).")
    parser.add_argument("--bookshelf", required=True, metavar="DIR",
                        help="Root directory of the ISPD 2005 bookshelf benchmarks. "
                             "Each chip is expected at DIR/<chip>/.")
    parser.add_argument("--our", action="store_true",
                        help="Evaluate our placer results from our/<chip>/.")
    parser.add_argument("--dreamplace", nargs="?", const=str(DREAMPLACE_ROOT), default=None,
                        metavar="DIR",
                        help="Evaluate DREAMPlace results from DIR/<chip>/<chip>_<method>.gp.pl "
                             "(default DIR: %s)" % DREAMPLACE_ROOT)
    parser.add_argument("--method", default="fft", metavar="METHOD",
                        help="DREAMPlace method suffix in filenames (default: fft). "
                             "One of: fft, mg, mgf.")
    parser.add_argument("--gp", action="store_true",
                        help="With --dreamplace: also compute HPWL on <chip>_<method>.gp.pl "
                             "and show as GP HPWL column. Legality only on final .dp.pl.")
    parser.add_argument("--row-window", type=int, default=LEGAL_ROW_WINDOW_DEFAULT,
                        metavar="N",
                        help="ROW_JUMP parameter for legal2.pl (default: %(default)s). "
                             "Larger values are faster but use more memory.")
    args = parser.parse_args()

    if args.dreamplace is None and not args.our:
        parser.error("Specify --dreamplace DIR or --our.")

    bookshelf_root = Path(args.bookshelf)
    row_window = args.row_window

    if args.dreamplace is not None:
        dreamplace_dir = Path(args.dreamplace)
        label = "DREAMPlace %s (%s)" % (args.method, dreamplace_dir)
    else:
        label = "our placer"
    print("Evaluating %s for all ISPD 2005 chips (row-window=%d)...\n"
          % (label, row_window))

    results = {}
    with ThreadPoolExecutor(max_workers=len(CHIPS)) as pool:
        if args.dreamplace is not None:
            futures = {pool.submit(evaluate_chip_dreamplace, chip, dreamplace_dir,
                                   args.method, bookshelf_root, row_window, args.gp): chip
                       for chip in CHIPS}
        else:
            futures = {pool.submit(evaluate_chip_our, chip, bookshelf_root, row_window): chip
                       for chip in CHIPS}
        for future in as_completed(futures):
            r = future.result()
            results[r["chip"]] = r
            status = r.get("error") or "HPWL=%s" % r["hpwl"]
            print("  [%s] done - %s" % (r["chip"], status))

    # -- Summary table -------------------------------------------------------
    if args.dreamplace is not None:
        if args.gp:
            hdr = "%-12s  %14s  %14s  %s"
            print("\n" + "=" * 78)
            print(hdr % ("Chip", "GP HPWL", "Final HPWL", "Legality"))
            print("-" * 78)
            for chip in CHIPS:
                r = results[chip]
                if r["error"]:
                    print("%-12s  %s" % (chip, "ERROR: %s" % r["error"]))
                else:
                    gp = ("{:,}".format(r["hpwl_gp"])) if r["hpwl_gp"] is not None else "N/A"
                    print(hdr % (chip, gp, "{:,}".format(r["hpwl"]), r["legal"]))
            print("=" * 78)
        else:
            print("\n" + "=" * 62)
            print("%-12s  %14s  %s" % ("Chip", "HPWL", "Legality"))
            print("-" * 62)
            for chip in CHIPS:
                r = results[chip]
                if r["error"]:
                    print("%-12s  %14s  ERROR: %s" % (chip, "N/A", r["error"]))
                else:
                    print("%-12s  %14s  %s" % (chip, "{:,}".format(r["hpwl"]), r["legal"]))
            print("=" * 62)
    else:
        hdr = "%-12s  %14s  %14s  %s"
        print("\n" + "=" * 78)
        print(hdr % ("Chip", "GP HPWL", "Final HPWL", "Legality"))
        print("-" * 78)
        for chip in CHIPS:
            r = results[chip]
            if r["error"]:
                print("%-12s  %s" % (chip, "ERROR: %s" % r["error"]))
            else:
                gp = ("{:,}".format(r["hpwl_gp"])) if r["hpwl_gp"] is not None else "N/A"
                print(hdr % (chip, gp, "{:,}".format(r["hpwl"]), r["legal"]))
        print("=" * 78)


if __name__ == "__main__":
    main()
