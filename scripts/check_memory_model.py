"""Recompute the memory-model numbers from results/arch_sweep/arch_sweep.json and check them
against the values the paper reports.

Cache term: allocated slots = ceil((L + D) / P) * P with (L, D, P) parsed from
experiments/arch_sweep.py (see src/memory_model.py for where that comes from).

COEFFICIENT CHOICE: integer, activation = 3 * d_model + 2 * d_ff elements per token. These are
the five tensors live at the peak (the GELU, src/model_core/mlp.py:22): the block input
(model.py:29), the post-attention residual (transformer_block.py:36/38), the LayerNorm-2 output
(transformer_block.py:39), and the fc1 and GELU outputs (mlp.py:21, :22). Counted on the GPU in
every swept shape by experiments/activation_census.py (results/activation_census/). The free fit
on three shapes (3.030, 1.997, c0 -0.018 KB) is printed as validation, not used for prediction.

INT8 cache ratio: int8 codes (1 B) plus one FP16 scale (2 B) per token per head per layer, for
each of K and V (src/kv_cache/int8_paged.py:23-27), against 2 B per element in FP16, with
head_dim = d_model / n_heads = 64: (64 + 2) / (2 * 64) = 33/64 = 0.515625.

Run:   python scripts/check_memory_model.py
       python scripts/check_memory_model.py --against path/to/rerun_arch_sweep.json
         compares a rerun's peaks with the committed ones, tolerance +/-1 MB per point
Exit status 1 if a reported value fails or a rerun is out of tolerance.
"""
import argparse
import json
import os
import sys
from fractions import Fraction

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import pandas as pd  # noqa: E402

from src.memory_model import allocated_slots, arch_sweep_config, predict_slopes  # noqa: E402

ARCH = os.path.join(REPO, "results", "arch_sweep", "arch_sweep.json")
EXP3 = os.path.join(REPO, "results", "precision_sweep", "exp3_precision_sweep.csv")
GPT2 = "12L/768d/3072ff"
COEFFS = "integer"
MB_TOL = 1.0        # memory values reproduce to within 1 MB across environments
HEAD_DIM = 512 // 8  # study model d_model // n_heads (ModelConfig.head_dim)

# (claim as printed, reported value, how to recompute from {shape: row}, decimals)
CLAIMS = [
    ("measured GPT-2 slope 53.078 KB/token", 53.078, lambda v: v[GPT2]["measured"], 3),
    ("activation 4L 11.01 KB/token", 11.01, lambda v: v["4L/512d/2048ff"]["act_kb"], 2),
    ("activation 8L 11.00 KB/token", 11.00, lambda v: v["8L/512d/2048ff"]["act_kb"], 2),
    ("activation 16L 11.04 KB/token", 11.04, lambda v: v["16L/512d/2048ff"]["act_kb"], 2),
    ("activation 2x d_model 14.03 KB/token", 14.03, lambda v: v["8L/1024d/2048ff"]["act_kb"], 2),
    ("activation 2x d_ff 18.99 KB/token", 18.99, lambda v: v["8L/512d/4096ff"]["act_kb"], 2),
    ("worst error 0.08% over six shapes", 0.08, lambda v: max(abs(r["error_pct"]) for r in v.values()), 2),
    ("GPT-2 predicted 53.062 KB/token", 53.062, lambda v: v[GPT2]["predicted"], 3),
    ("GPT-2 error 0.03%", 0.03, lambda v: abs(v[GPT2]["error_pct"]), 2),
]


def show(name, coeffs_tuple, rows):
    c0, a, b = coeffs_tuple
    print(f"== {name}: activation = {a:.4f}*d_model + {b:.4f}*d_ff elements"
          f"{f' {c0:+.4f} KB' if c0 else ''}")
    print(f"{'shape':>16} {'held':>5} {'measured':>9} {'cache':>8} {'act':>8} "
          f"{'act pred':>8} {'predicted':>10} {'err %':>7}")
    for r in rows:
        print(f"{r['shape']:>16} {str(r['held_out']):>5} {r['measured']:9.3f} {r['cache_kb']:8.4f} "
              f"{r['act_kb']:8.4f} {r['act_pred_kb']:8.4f} {r['predicted']:10.4f} {r['error_pct']:+7.3f}")
    print(f"worst |error|: {max(abs(r['error_pct']) for r in rows):.3f}%\n")


def compare_rerun(path):
    """Peaks of a rerun vs the committed file, per architecture and point, +/-MB_TOL."""
    old = {r["label"]: r for r in json.load(open(ARCH))}
    bad = 0
    for r in json.load(open(path)):
        o = old.get(r["label"])
        if o is None or r["tokens"] != o["tokens"]:
            print(f"  SKIP  {r['label']}: not in committed file or different token counts")
            continue
        for t, new, ref in zip(r["tokens"], r["peaks"], o["peaks"]):
            ok = abs(new - ref) <= MB_TOL
            bad += not ok
            print(f"  {'ok  ' if ok else 'FAIL'}  {r['label']:<30} tokens {t:>7}  "
                  f"rerun {new:9.2f}  committed {ref:9.2f}  diff {new - ref:+.2f} MB")
    return bad


def int8_ratio():
    """Analytic INT8/FP16 cache ratio, and the measured range from the precision sweep."""
    analytic = Fraction(HEAD_DIM * 1 + 2, HEAD_DIM * 2)
    d = pd.read_csv(EXP3)
    d = d[d.status == "ok"].set_index(["seq_len", "batch"])
    r = d[d.cache_type == "paged_int8"].cache_memory_mb / d[d.cache_type == "paged_fp16"].cache_memory_mb
    print(f"INT8/FP16 cache ratio: analytic ({HEAD_DIM} + 2) / (2 * {HEAD_DIM}) = {analytic} = "
          f"{float(analytic):.6f}; measured {r.min():.4f}-{r.max():.4f} over {len(r)} cells "
          f"(cache_memory_mb is stored to 0.01 MB) -> {r.mean():.3f}")
    return analytic, r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--against", help="a rerun arch_sweep.json to compare with the committed one")
    args = ap.parse_args()

    arch = json.load(open(ARCH))
    L, D, P = arch_sweep_config()
    slots = allocated_slots(L, D, P)
    print(f"arch_sweep config L={L} D={D} P={P}: allocated slots ceil(({L}+{D})/{P})*{P} = {slots}"
          f"  (x{slots}/{L} = {slots / L:.6f})\n")

    cf, chosen = predict_slopes(arch, corrected=True, coeffs=COEFFS)
    show(f"{COEFFS} coefficients, corrected cache term (CHOSEN)", cf, chosen)
    _, uncorr = predict_slopes(arch, corrected=False, coeffs=COEFFS)
    show(f"{COEFFS} coefficients, uncorrected cache term (for comparison)", (0.0, 3.0, 2.0), uncorr)
    ff, fitted = predict_slopes(arch, corrected=True, coeffs="fitted")
    show("VALIDATION: free fit on 8L/512d/2048ff, 8L/1024d/2048ff, 8L/512d/4096ff", ff, fitted)
    print(f"Free fit recovers a = {ff[1]:.3f} (integer 3), b = {ff[2]:.3f} (integer 2), "
          f"c0 = {ff[0]:+.3f} KB (integer 0)\n")

    v = {r["shape"]: r for r in chosen}
    failed = 0
    print(f"Reported values ({COEFFS} coefficients, corrected cache term):")
    for claim, reported, fn, nd in CLAIMS:
        got = fn(v)
        ok = round(got, nd) == reported
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {claim:<40} recomputed {got:.4f}")
    for claim, reported, got in [("fit a = 3.03", 3.03, ff[1]), ("fit b = 2.00", 2.00, ff[2])]:
        ok = round(got, 2) == reported
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {claim:<40} recomputed {got:.4f}")
    print(f"  info  GPT-2 error with uncorrected cache term: "
          f"{abs({r['shape']: r for r in uncorr}[GPT2]['error_pct']):.2f}%\n")

    analytic, r = int8_ratio()
    ok = analytic == Fraction(33, 64) and round(r.mean(), 3) == 0.516
    failed += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  INT8 ratio 33/64 = 0.515625 analytic, 0.516 measured")

    if args.against:
        print(f"\nRerun comparison, tolerance +/-{MB_TOL:g} MB: {args.against}")
        failed += compare_rerun(args.against)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
