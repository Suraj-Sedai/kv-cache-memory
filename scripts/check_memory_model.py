"""Recompute the memory-model numbers from results/arch_sweep/arch_sweep.json and check them
against the values the paper reports.

The cache term uses allocated slots = ceil((L + D) / P) * P with (L, D, P) parsed from
experiments/arch_sweep.py (see src/memory_model.py for where that comes from).

COEFFICIENT CHOICE: pending. Both activation-coefficient sets are printed and checked:
  integer  3 * d_model + 2 * d_ff        (paper Table 1)
  fitted   solved on three shapes        (notebook 05)

Run:   python scripts/check_memory_model.py
       python scripts/check_memory_model.py --against path/to/rerun_arch_sweep.json
         compares a rerun's peaks with the committed ones, tolerance +/-1 MB per point
Exit status 1 if a reported value fails under every coefficient set, or a rerun is out of tolerance.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.memory_model import allocated_slots, arch_sweep_config, predict_slopes  # noqa: E402

ARCH = os.path.join(REPO, "results", "arch_sweep", "arch_sweep.json")
GPT2 = "12L/768d/3072ff"
MB_TOL = 1.0        # memory values reproduce to within 1 MB across environments

# (claim as printed, how to recompute it from a {variant: {shape: row}} result, decimals)
CLAIMS = [
    ("measured GPT-2 slope 53.078 KB/token", 53.078, lambda v: v[GPT2]["measured"], 3),
    ("activation 4L 11.01 KB/token", 11.01, lambda v: v["4L/512d/2048ff"]["act_kb"], 2),
    ("activation 8L 11.00 KB/token", 11.00, lambda v: v["8L/512d/2048ff"]["act_kb"], 2),
    ("activation 16L 11.04 KB/token", 11.04, lambda v: v["16L/512d/2048ff"]["act_kb"], 2),
    ("worst error 0.08% over six shapes", 0.08, lambda v: max(abs(r["error_pct"]) for r in v.values()), 2),
    ("GPT-2 predicted 53.062 KB/token", 53.062, lambda v: v[GPT2]["predicted"], 3),
    ("GPT-2 error 0.03%", 0.03, lambda v: abs(v[GPT2]["error_pct"]), 2),
]


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--against", help="a rerun arch_sweep.json to compare with the committed one")
    args = ap.parse_args()

    arch = json.load(open(ARCH))
    L, D, P = arch_sweep_config()
    slots = allocated_slots(L, D, P)
    print(f"arch_sweep config L={L} D={D} P={P}: allocated slots ceil(({L}+{D})/{P})*{P} = {slots}"
          f"  (x{slots}/{L} = {slots / L:.6f})\n")

    variants = {}
    for coeffs in ("integer", "fitted"):
        for corrected in (False, True):
            (c0, a, b), rows = predict_slopes(arch, corrected=corrected, coeffs=coeffs)
            name = f"{coeffs}, {'corrected' if corrected else 'uncorrected'}"
            variants[name] = {r["shape"]: r for r in rows}
            print(f"== {name}: activation = {a:.4f}*d_model + {b:.4f}*d_ff elements"
                  f"{f' {c0:+.4f} KB' if c0 else ''}")
            print(f"{'shape':>16} {'held':>5} {'measured':>9} {'cache':>8} {'act':>8} "
                  f"{'predicted':>10} {'err %':>7}")
            for r in rows:
                print(f"{r['shape']:>16} {str(r['held_out']):>5} {r['measured']:9.3f} "
                      f"{r['cache_kb']:8.4f} {r['act_kb']:8.4f} {r['predicted']:10.4f} "
                      f"{r['error_pct']:+7.3f}")
            print(f"worst |error|: {max(abs(r['error_pct']) for r in rows):.3f}%\n")

    corrected = [k for k in variants if k.endswith(", corrected")]
    failed = 0
    print("Reported values (corrected cache term):")
    for claim, reported, fn, nd in CLAIMS:
        res = {k.split(",")[0]: fn(variants[k]) for k in corrected}
        passing = [k for k, v in res.items() if round(v, nd) == reported]
        failed += not passing
        detail = "  ".join(f"{k} {v:.4f}" for k, v in res.items())
        print(f"  {'PASS' if passing else 'FAIL'}  {claim:<40} {detail}"
              f"   passes with: {', '.join(passing) or 'none'}")
    u = variants["fitted, uncorrected"][GPT2]["error_pct"]
    print(f"  info  GPT-2 error with uncorrected cache term (fitted): {abs(u):.2f}%")

    if args.against:
        print(f"\nRerun comparison, tolerance +/-{MB_TOL:g} MB: {args.against}")
        failed += compare_rerun(args.against)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
