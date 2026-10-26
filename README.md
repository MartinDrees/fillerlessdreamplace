# Reproducing Results from "Simple Electrostatic Global Placement without Filler Cells"

This repository contains the code needed to reproduce the results from the paper.
It is built on top of [DREAMPlace](https://github.com/limbo018/DREAMPlace) with modifications that add a fillerless multigrid electrostatic potential.

## 1. Precomputed Results: Logfiles, Bookshelf Placements, and Wirelength Verification

The `paper_results/` directory contains all outputs needed to verify the reported wirelengths **without re-running the placer**.

### Placement outputs (bookshelf `.pl` files)

| Subdirectory | Methods | Files per benchmark |
|---|---|---|
| `paper_results/dreamplace/<benchmark>/` | `fft`, `mg`, `mgf` | `<benchmark>_<method>.gp.pl` (global placement), `<benchmark>_<method>.dp.pl` (detailed placement) |
| `paper_results/our/<benchmark>/` | `our` | `<benchmark>_our.gp.pl`, `<benchmark>_our.dp.pl` |

All eight ISPD 2005 benchmarks (adaptec1–4, bigblue1–4) are covered.

### Log files

DREAMPlace log files for the `fft`, `mg`, and `mgf` runs are stored alongside the placement outputs:

```
paper_results/dreamplace/<benchmark>/<benchmark>_<method>.log
```

These logs contain the HPWL figures reported internally by DREAMPlace and are summarised in `paper_results/ispd2005.csv`.

### Verifying the reported wirelengths

The paper reports HPWL numbers produced by the **official ISPD 2005 evaluation scripts** (`hpwl.pl`, `legal2.pl`), which are included in `paper_results/`. To recompute those numbers from the stored placement outputs, first download the benchmarks:

```bash
cd paper_results
bash download_ispd2005.sh
cd ..
```

This creates `paper_results/bookshelf/` containing the unpacked bookshelf files for all eight benchmarks (adaptec1–4, bigblue1–4). Then run:

```bash
cd paper_results
python3 run_evaluation.py --bookshelf bookshelf
```

This invokes `evaluate_ispd.py` for all four methods (`fft`, `mg`, `mgf`, `our`) in parallel and prints the HPWL and legality result for each benchmark, matching the numbers in the paper.

> **Note:** The HPWL figures in `ispd2005.csv` are taken from DREAMPlace's internal log output and differ slightly from the official evaluation script results. The paper uses the official evaluation script numbers; the CSV is provided for reference only.

---

## 2. Build

Follow the standard DREAMPlace build instructions (see `README-DREAMPlace.md`). Note that minor compilation changes may be necessary depending on your local environment.

## 3. Run the Comparison

`run_comparison.py` runs DREAMPlace on all ISPD 2005 instances in three modes and prints a summary table:

| Mode  | Description                                         |
|-------|-----------------------------------------------------|
| `fft` | Standard FFT-based electrostatic potential (with filler cells) |
| `mg`  | Fillerless multigrid electrostatic potential         |
| `mgf` | Multigrid electrostatic potential with filler cells  |

Results are written to `results/comparison/` (logs) and summarised in `results/comparison/ispd2005.csv`.
