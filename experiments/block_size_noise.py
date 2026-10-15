"""Is the block-size throughput spread real on a single commit?

The withdrawn block-size claim rests on one cell (paged, seq 1100, batch 16, block 32)
measured at 226.6 tok/s (13fbfae) and 495.3 tok/s (e383030). The TPOT fix 0d029d6 sits
between those commits, so they are not the same code. This runs the same cell N times
back-to-back on the current commit, each run a fresh process of run_block_size_sweep.py,
and reports the spread.

Default is the single cell. --full reruns the whole 60-cell sweep each time (slow), which is
closer to how the original numbers were produced (thermal state from earlier cells).

Run:  python -m experiments.block_size_noise
      writes results/block_size_sweep/noise_run{i}.csv and noise_check.json
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

OUT_DIR = os.path.join("results", "block_size_sweep")
CELL = dict(block_size=32, seq_len=1100, batch=16)
HISTORICAL = {"13fbfae": 226.6, "e383030": 495.3}


def main():
    ap = argparse.ArgumentParser(description="Back-to-back reruns of one block-size cell.")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="rerun the full sweep each time")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    runs = []
    for i in range(1, args.runs + 1):
        out = os.path.join(args.out_dir, f"noise_run{i}.csv")
        cmd = [sys.executable, "-m", "experiments.run_block_size_sweep",
               "--seed", str(args.seed), "--output", out]
        if not args.full:
            cmd += ["--block-sizes", str(CELL["block_size"]), "--seq-lens", str(CELL["seq_len"]),
                    "--batches", str(CELL["batch"])]
        print(f"run {i}: {' '.join(cmd)}")
        t0 = time.time()
        subprocess.run(cmd, check=True)
        with open(out, newline="") as f:
            row = next(r for r in csv.DictReader(f)
                       if all(int(r[k]) == v for k, v in CELL.items()))
        runs.append(dict(run=i, csv=out, wall_s=round(time.time() - t0, 1),
                         **{k: row[k] for k in ("status", "throughput", "throughput_cv",
                                                "tpot_ms", "peak_memory_mb")}))
        print(runs[-1])

    tput = [float(r["throughput"]) for r in runs if r["status"] == "ok"]
    ratio = max(tput) / min(tput) if len(tput) >= 2 else None
    hist_ratio = max(HISTORICAL.values()) / min(HISTORICAL.values())
    summary = dict(cell=CELL, commit=commit, seed=args.seed, full_sweep=args.full, runs=runs,
                   max_over_min=round(ratio, 3) if ratio else None,
                   historical=HISTORICAL, historical_max_over_min=round(hist_ratio, 3),
                   note="Paper claim: 'Two runs of the same config gave 227 and 495 tok/s'. "
                        "If max_over_min on one commit is far below the historical ratio, the "
                        "claim cannot be supported as 'same config' and should be removed.")
    print(f"\nmax/min throughput on {commit}: {summary['max_over_min']}   "
          f"historical (two commits): {summary['historical_max_over_min']}")
    path = os.path.join(args.out_dir, "noise_check.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
