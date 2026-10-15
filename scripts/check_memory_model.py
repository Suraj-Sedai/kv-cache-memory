"""Recompute the memory-model numbers from results/arch_sweep/arch_sweep.json,
uncorrected and corrected, and check them against the values the paper reports.

Run:  python scripts/check_memory_model.py      (exit 1 if any reported value fails)
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.memory_model import ARCH_SWEEP_POOL, pool_correction, predict_slopes  # noqa: E402

ARCH = os.path.join(REPO, "results", "arch_sweep", "arch_sweep.json")
GPT2 = "12L/768d/3072ff"


def main():
    arch = json.load(open(ARCH))
    c = pool_correction(**ARCH_SWEEP_POOL)
    print(f"pool correction = {c * 1024:.0f}/1024 = {c:.6f}\n")

    res = {}
    for label, corr in (("uncorrected", 1.0), ("corrected", c)):
        coeffs, rows = predict_slopes(arch, corr)
        res[label] = {r["shape"]: r for r in rows}
        print(f"== {label}: activation = {coeffs[0]:+.4f} {coeffs[1]:+.6f}*d_model "
              f"{coeffs[2]:+.6f}*d_ff")
        print(f"{'shape':>16} {'held':>5} {'measured':>9} {'cache':>8} {'act':>8} "
              f"{'predicted':>10} {'err %':>7}")
        for r in rows:
            print(f"{r['shape']:>16} {str(r['held_out']):>5} {r['measured']:9.3f} "
                  f"{r['cache_kb']:8.3f} {r['act_kb']:8.4f} {r['predicted']:10.3f} "
                  f"{r['error_pct']:7.3f}")
        worst = max(abs(r["error_pct"]) for r in rows)
        print(f"worst |error| over {len(rows)} shapes: {worst:.3f}%\n")

    u, k = res["uncorrected"], res["corrected"]
    checks = [
        # (claim as reported, recomputed value, rounding used in the paper)
        ("measured GPT-2 slope 53.078 KB/token", k[GPT2]["measured"], 3, 53.078),
        ("activation 4L 11.01 KB/token (corrected)", k["4L/512d/2048ff"]["act_kb"], 2, 11.01),
        ("activation 8L 11.00 KB/token (corrected)", k["8L/512d/2048ff"]["act_kb"], 2, 11.00),
        ("activation 16L 11.04 KB/token (corrected)", k["16L/512d/2048ff"]["act_kb"], 2, 11.04),
        ("worst error 0.08% over six shapes (corrected)",
         max(abs(r["error_pct"]) for r in k.values()), 2, 0.08),
        ("GPT-2 error 0.37% (uncorrected)", abs(u[GPT2]["error_pct"]), 2, 0.37),
        ("GPT-2 predicted 53.06 KB/token (corrected)", k[GPT2]["predicted"], 2, 53.06),
        ("GPT-2 error 0.03% (corrected)", abs(k[GPT2]["error_pct"]), 2, 0.03),
    ]
    failed = 0
    for claim, value, nd, reported in checks:
        ok = round(value, nd) == reported
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {claim:<48} recomputed {value:.4f}")
    if failed:
        print(f"\n{failed} reported value(s) do not reproduce from arch_sweep.json.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
