"""Print the README / paper tables from the committed results. No GPU needed.

  table 1  cache share of peak and INT8 peak reduction    <- results/precision_sweep/exp3_precision_sweep.csv
  table 2  FP16 / INT8 per-token fits, tokens that fit in 8 GB (same file; notebook 04's fit)
  table 3  INT8 perplexity cost on real GPT-2 weights      <- results/perplexity_eval/exp3_perplexity.csv
  table 4  paged / contiguous throughput range             <- results/cache_comparison/exp1_layout_comparison.csv

The memory-model numbers (0.08%, 11.00 KB/token, ...) are in scripts/check_memory_model.py.

Memory values reproduce to within MB_TOL = 1 MB across environments, so reported MB values and
rerun comparisons are checked at +/-1 MB; percentages are checked at their printed precision.

Run:  python scripts/make_tables.py
      python scripts/make_tables.py --against path/to/rerun.csv
        compares peak/cache memory of a rerun CSV with the committed CSV of the same name, +/-1 MB
Exit status 1 if a reported value or a rerun comparison fails.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(REPO, "results")
MB_TOL = 1.0
KEYS = ["cache_type", "seq_len", "batch", "block_size"]
MEM = ["peak_memory_mb", "cache_memory_mb"]


def table1_2():
    d = pd.read_csv(os.path.join(R, "precision_sweep", "exp3_precision_sweep.csv"))
    d = d[d.status == "ok"]
    f = d[d.cache_type == "paged_fp16"].set_index(["seq_len", "batch"])
    q = d[d.cache_type == "paged_int8"].set_index(["seq_len", "batch"])
    t = pd.DataFrame({
        "fp16_cache_mb": f.cache_memory_mb, "fp16_peak_mb": f.peak_memory_mb,
        "cache_share_pct": 100 * f.cache_memory_mb / f.peak_memory_mb,
        "int8_peak_mb": q.peak_memory_mb,
        "int8_cuts_peak_pct": 100 * (f.peak_memory_mb - q.peak_memory_mb) / f.peak_memory_mb,
        "int8_cache_smaller_pct": 100 * (1 - q.cache_memory_mb / f.cache_memory_mb),
    })
    print("Table 1: cache share of peak and INT8 savings (README rows marked *)")
    readme = {(512, 1), (1024, 8), (2048, 16), (8192, 32)}
    t.insert(0, "readme", ["*" if i in readme else "" for i in t.index])
    print(t.round(1).to_string())
    print(f"  INT8 cache smaller by {t.int8_cache_smaller_pct.min():.1f}-{t.int8_cache_smaller_pct.max():.1f}%;"
          f" cache share {t.cache_share_pct.min():.1f}-{t.cache_share_pct.max():.1f}%;"
          f" INT8 peak cut {t.int8_cuts_peak_pct.min():.1f}-{t.int8_cuts_peak_pct.max():.1f}%\n")

    print("Table 2: peak_MB = slope * (batch*seq_len) + intercept (numpy polyfit, as notebook 04)")
    fits = {}
    for dt in ("fp16", "int8"):
        s = d[d.cache_type == f"paged_{dt}"]
        slope, icept = np.polyfit(s.seq_len * s.batch, s.peak_memory_mb, 1)
        fits[dt] = (slope * 1024, icept)
        fit8 = (8 * 1024 - icept) / slope
        print(f"  {dt}: {slope * 1024:.2f} KB/token + {icept:.0f} MB   tokens in 8 GB: {fit8:,.0f}")
    t8 = {dt: (8 * 1024 - i) / (s / 1024) for dt, (s, i) in fits.items()}
    print(f"  INT8 fits {100 * (t8['int8'] / t8['fp16'] - 1):.1f}% more tokens at 8 GB\n")


def table3():
    p = pd.read_csv(os.path.join(R, "perplexity_eval", "exp3_perplexity.csv"))
    cited = p[p.excluded.isna()]
    print("Table 3: INT8 KV perplexity cost (excluded rows not shown)")
    print(cited.drop(columns=["excluded"]).to_string(index=False))
    print(f"  |PPL change| <= {cited.ppl_pct_increase.abs().max():.3f}%;"
          f" top-1 predictions flipped {100 * (1 - cited.top1_agreement.max()):.1f}-"
          f"{100 * (1 - cited.top1_agreement.min()):.1f}%;"
          f" {len(p) - len(cited)} excluded row(s)\n")


def table4():
    d = pd.read_csv(os.path.join(R, "cache_comparison", "exp1_layout_comparison.csv"))
    d = d[d.status == "ok"]
    m = d[d.cache_type == "paged"].set_index(["seq_len", "batch"]).join(
        d[d.cache_type == "contiguous"].set_index(["seq_len", "batch"]), lsuffix="_p", rsuffix="_c")
    ratio = m.throughput_p / m.throughput_c
    peak = 100 * (m.peak_memory_mb_p / m.peak_memory_mb_c - 1)
    print("Table 4: paged / contiguous")
    print(f"  throughput ratio {ratio.min():.2f}-{ratio.max():.2f}x over {len(m)} cells;"
          f" paged > contiguous in {(ratio > 1).sum()} cells;"
          f" peak memory differs by at most {peak.abs().max():.1f}%")


def check_reported():
    """Values the paper prints, recomputed from results/precision_sweep (decode length 60)."""
    d = pd.read_csv(os.path.join(R, "precision_sweep", "exp3_precision_sweep.csv")).set_index(
        ["cache_type", "seq_len", "batch"])
    f = lambda s, b, c: d.loc[("paged_fp16", s, b), c]
    q = lambda s, b, c: d.loc[("paged_int8", s, b), c]
    share = lambda s, b: 100 * f(s, b, "cache_memory_mb") / f(s, b, "peak_memory_mb")
    cut = lambda s, b: 100 * (f(s, b, "peak_memory_mb") - q(s, b, "peak_memory_mb")) / f(s, b, "peak_memory_mb")
    checks = [  # (claim, recomputed, reported, tolerance)
        ("512x1 cache 9 MB", f(512, 1, "cache_memory_mb"), 9, MB_TOL),
        ("512x1 peak 172 MB", f(512, 1, "peak_memory_mb"), 172, MB_TOL),
        ("8192x32 cache 4128 MB", f(8192, 32, "cache_memory_mb"), 4128, MB_TOL),
        ("8192x32 peak 7113 MB", f(8192, 32, "peak_memory_mb"), 7113, MB_TOL),
        ("2048x16 post-fix peak 1039 MB", f(2048, 16, "peak_memory_mb"), 1039, MB_TOL),
        ("share 512x1 5.2%", share(512, 1), 5.2, 0.05),
        ("share 1024x8 35.5%", share(1024, 8), 35.5, 0.05),
        ("share 2048x16 50.8%", share(2048, 16), 50.8, 0.05),
        ("share 8192x32 58.0%", share(8192, 32), 58.0, 0.05),
        ("INT8 peak cut 512x1 2.7%", cut(512, 1), 2.7, 0.05),
        ("INT8 peak cut 1024x8 15.9%", cut(1024, 8), 15.9, 0.05),
        ("INT8 peak cut 2048x16 22.8%", cut(2048, 16), 22.8, 0.05),
        ("INT8 peak cut 8192x32 26.0%", cut(8192, 32), 26.0, 0.05),
    ]
    print("Reported values")
    bad = 0
    for claim, got, rep, tol in checks:
        ok = abs(got - rep) <= tol
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {claim:<32} recomputed {got:.3f}")
    return bad


def compare(path):
    """Rerun CSV vs the committed CSV with the same file name, memory columns, +/-MB_TOL."""
    name = os.path.basename(path)
    ref = [p for p in glob.glob(os.path.join(R, "*", name)) if "pre_fix" not in p]
    if len(ref) != 1:
        sys.exit(f"no unique committed results/*/{name} to compare against")
    new, old = pd.read_csv(path, dtype={"block_size": str}), pd.read_csv(ref[0], dtype={"block_size": str})
    keys = [k for k in KEYS if k in new and k in old]
    m = new.merge(old, on=keys, suffixes=("_new", "_old"))
    print()
    print(f"Rerun {path} vs {os.path.relpath(ref[0], REPO)}: {len(m)} matching cells, tolerance +/-{MB_TOL:g} MB")
    bad = 0
    for _, r in m.iterrows():
        for c in MEM:
            a, b = r[c + "_new"], r[c + "_old"]
            if pd.isna(a) and pd.isna(b):
                continue
            ok = abs(a - b) <= MB_TOL
            bad += not ok
            print(f"  {'ok  ' if ok else 'FAIL'}  {tuple(r[k] for k in keys)}  {c:<16} "
                  f"rerun {a:9.2f}  committed {b:9.2f}  diff {a - b:+.2f}")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Print the paper tables from committed results.")
    ap.add_argument("--against", nargs="*", default=[], help="rerun CSV(s) to compare with the committed ones")
    args = ap.parse_args()
    table1_2()
    table3()
    table4()
    print()
    failed = check_reported()
    for p in args.against:
        failed += compare(p)
    sys.exit(1 if failed else 0)
