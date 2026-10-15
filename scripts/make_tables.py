"""Print the README / paper tables from the committed results. No GPU needed.

  table 1  cache share of peak and INT8 peak reduction    <- results/precision_sweep/exp3_precision_sweep.csv
  table 2  FP16 / INT8 per-token fits, tokens that fit in 8 GB (same file; notebook 04's fit)
  table 3  INT8 perplexity cost on real GPT-2 weights      <- results/perplexity_eval/exp3_perplexity.csv
  table 4  paged / contiguous throughput range             <- results/cache_comparison/exp1_layout_comparison.csv

The memory-model numbers (0.08%, 11.00 KB/token, ...) are in scripts/check_memory_model.py.

Run:  python scripts/make_tables.py
"""
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(REPO, "results")


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


if __name__ == "__main__":
    table1_2()
    table3()
    table4()
