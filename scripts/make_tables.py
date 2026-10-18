"""Print the README / paper tables from the committed results, and write the paper's numeric
tables as LaTeX (booktabs) to paper/tables/. No GPU needed.

  paper/tables/table1.tex   paper Table 1, per-token memory predicted vs measured
                            <- results/arch_sweep/arch_sweep.json (same computation as
                               scripts/check_memory_model.py); checked cell by cell against
                               paper.md when that local file is present
  (Paper Table 2, the four measurement defects, is prose with numbers traced in paper/NUMBERS.md.)

Printed below (numbered by this script, not the paper):

  table 1  cache share of peak and INT8 peak reduction    <- results/precision_sweep/exp3_precision_sweep.csv
  table 2  FP16 / INT8 per-token fits, tokens that fit in 8 GB (same file; notebook 04's fit)
  table 3  INT8 perplexity cost on real GPT-2 weights      <- results/perplexity_eval/exp3_perplexity.csv
  table 4  paged / contiguous throughput range             <- results/cache_comparison/exp1_layout_comparison.csv
  table 5  decode-step FLOPs and arithmetic share of TPOT  <- results/precision_sweep, results/matmul_peak
  table 6  pre-fix ledger at 2048x16 and across the grid  <- results/pre_fix/exp3_precision_sweep_017b63a.csv
  table 7  effective ceiling below each enforced budget    <- results/oom_frontier, results/pre_fix/wddm_spill.json
  table 8  withdrawn block-size ranking, cells won         <- results/pre_fix/exp2_*, results/block_size_sweep
  table 9  pitfall-2 batch effect; interleaved vs unpaired <- results/pre_fix/exp1_*, results/paired_layout_check, exp1
  table 10 paged read() gather vs the paged-contiguous gap <- results/gather_cost/gather_cost.json (5 reps), _r15 (15 reps)

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
import json
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")       # mismatch reports print the paper's U+2212
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(REPO, "results")
TABLES = os.path.join(REPO, "paper", "tables")
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


def decode_step_flops(n_layers, d_model, d_ff, vocab, batch, context):
    """FLOPs of one decode step (one new token per sequence), multiply-add = 2 FLOPs.

    Inputs and where they come from:
      n_layers, d_model, d_ff  N_LAYERS, D_MODEL, D_FF in experiments/run_precision_sweep.py
      vocab                    ModelConfig.vocab_size (50257)
      batch, context           the CSV cell: batch, and seq_len + tokens already decoded
    Per sequence: projections 2*n_layers*(4*d_model^2 + 2*d_model*d_ff), attention over the
    context 4*n_layers*d_model*context (QK^T and AV), LM head 2*d_model*vocab.
    """
    per_seq = (2 * n_layers * (4 * d_model ** 2 + 2 * d_model * d_ff)
               + 4 * n_layers * d_model * context + 2 * d_model * vocab)
    return batch * per_seq


def table5():
    """Arithmetic share of a decode step = FLOPs / achievable FP16 throughput / measured TPOT.

    Throughput comes from results/matmul_peak/matmul_peak.json (python -m experiments.matmul_peak);
    every other input is in committed files. Reported per cell, with the maximum over the sweep
    (an upper bound for "at this model size") and the 2048x16 cell used throughout section 4.
    """
    d = pd.read_csv(os.path.join(R, "precision_sweep", "exp3_precision_sweep.csv"))
    d = d[(d.status == "ok") & (d.cache_type == "paged_fp16")].copy()
    decode_len = 60                                      # DECODE_LEN in run_precision_sweep.py
    d["context"] = d.seq_len + decode_len / 2            # mean context over the decode steps
    d["gflop_per_step"] = [decode_step_flops(8, 512, 2048, 50257, r.batch, r.context) / 1e9
                           for r in d.itertuples()]
    print("Table 5: decode-step arithmetic (FP16 paged, precision sweep)")
    cols = ["seq_len", "batch", "tpot_ms", "gflop_per_step"]
    peak_path = os.path.join(R, "matmul_peak", "matmul_peak.json")
    if os.path.exists(peak_path):
        tflops = json.load(open(peak_path))["achievable_tflops"]
        d["arith_ms"] = d.gflop_per_step / tflops      # 1 TFLOP/s = 1 GFLOP per ms
        d["arith_pct_of_tpot"] = 100 * d.arith_ms / d.tpot_ms
        cols += ["arith_ms", "arith_pct_of_tpot"]
        top = d.loc[d.arith_pct_of_tpot.idxmax()]
        ref = d[(d.seq_len == 2048) & (d.batch == 16)].iloc[0]
        print(f"  achievable FP16 {tflops:g} TFLOP/s ({os.path.relpath(peak_path, REPO)}): arithmetic is "
              f"{d.arith_pct_of_tpot.min():.2f}-{d.arith_pct_of_tpot.max():.2f}% of TPOT; max at "
              f"{int(top.seq_len)}x{int(top.batch)}; 2048x16: {ref.arith_pct_of_tpot:.2f}%")
    else:
        print("  results/matmul_peak/matmul_peak.json not found: run python -m experiments.matmul_peak")
    print(d[cols].round(4).to_string(index=False))


def table6():
    """Pre-fix ledger (LM head over all prefill positions), single source exp3 @017b63a."""
    pre = pd.read_csv(os.path.join(R, "pre_fix", "exp3_precision_sweep_017b63a.csv"))
    post = pd.read_csv(os.path.join(R, "precision_sweep", "exp3_precision_sweep.csv"))
    pick = lambda d, s, b: d[(d.cache_type == "paged_fp16") & (d.seq_len == s) & (d.batch == b)].iloc[0]
    a, b = pick(pre, 2048, 16), pick(post, 2048, 16)
    logits_mb = 16 * 2048 * 50257 * 2 / 2 ** 20              # [B, T, vocab] fp16, all positions
    print("Table 6: LM-head fix at 2048x16, FP16 paged (pre: results/pre_fix/exp3_precision_sweep_017b63a.csv)")
    print(f"  before: cache {a.cache_memory_mb:.1f} / peak {a.peak_memory_mb:.2f} MB = "
          f"{100 * a.cache_memory_mb / a.peak_memory_mb:.1f}%; logits 16*2048*50257*2 B = {logits_mb:.1f} MB "
          f"= {100 * logits_mb / a.peak_memory_mb:.1f}% of peak")
    print(f"  after:  cache {b.cache_memory_mb:.1f} / peak {b.peak_memory_mb:.2f} MB = "
          f"{100 * b.cache_memory_mb / b.peak_memory_mb:.1f}%")
    g = pre[(pre.cache_type == "paged_fp16") & (pre.status == "ok")].copy()
    g["share"] = 100 * g.cache_memory_mb / g.peak_memory_mb
    lo, hi = g.loc[g.share.idxmin()], g.loc[g.share.idxmax()]
    print(f"  before, across the {len(g)} ok FP16 cells: {lo.share:.1f}% ({int(lo.seq_len)}x{int(lo.batch)}) "
          f"to {hi.share:.1f}% ({int(hi.seq_len)}x{int(hi.batch)}); not ok: "
          + ", ".join(f"{int(r.seq_len)}x{int(r.batch)} {r.status}"
                      for r in pre[(pre.cache_type == "paged_fp16") & (pre.status != "ok")].itertuples()))


def table7():
    """Effective ceiling under set_per_process_memory_fraction, bracketed by the data.

    The cap applies to reserved memory, the peak column is allocated memory. For each budget:
    a cell that completes proves the ceiling is >= its allocated peak; a cell that OOMs although
    its allocated peak (known from another budget, or from the uncapped wddm_spill run) is under
    the budget proves the ceiling is < that peak.
    """
    o = json.load(open(os.path.join(R, "oom_frontier", "oom_frontier.json")))
    w = json.load(open(os.path.join(R, "pre_fix", "wddm_spill.json")))["cells"]
    peak = {(r["seq_len"], r["batch"], dt): r[dt]["peak_mb"] for r in o for dt in ("fp16", "int8")
            if r[dt]["status"] == "ok"}
    for c in w:
        peak.setdefault((c["seq_len"], c["batch"], c["dtype"]), c["peak_allocated_mb"])
    print("Table 7: effective ceiling vs stated budget (results/oom_frontier + results/pre_fix/wddm_spill.json)")
    for B in sorted({r["budget_gb"] for r in o}):
        cap, lo, hi = B * 1024, (0.0, None), (None, None)
        for r in (r for r in o if r["budget_gb"] == B):
            for dt in ("fp16", "int8"):
                k = (r["seq_len"], r["batch"], dt)
                P = peak.get(k)
                if P is None:
                    continue
                if r[dt]["status"] == "ok" and P > lo[0]:
                    lo = (P, k)
                if r[dt]["status"] == "oom" and P < cap and (hi[0] is None or P < hi[0]):
                    hi = (P, k)
        name = lambda k: f"{k[0]}x{k[1]} {k[2]}"
        line = f"  {B:>4g} GB: ceiling >= {lo[0]:.1f} MB ({name(lo[1])}), at most {100 * (1 - lo[0] / cap):.1f}% below"
        if hi[0] is not None:
            line += f"; ceiling < {hi[0]:.1f} MB ({name(hi[1])}), at least {100 * (1 - hi[0] / cap):.1f}% below"
        else:
            line += "; no OOM below the budget, so no lower bound on the gap"
        print(line)


def table8():
    """Withdrawn block-size ranking: which block size wins each (seq_len, batch) cell."""
    print("Table 8: block size with the highest throughput per (seq_len, batch) cell (withdrawn claim)")
    for name, path in [("13fbfae (the ranking the paper withdraws)", ("pre_fix", "exp2_block_size_sweep_13fbfae.csv")),
                       ("017b63a", ("pre_fix", "exp2_block_size_sweep_017b63a.csv")),
                       ("current", ("block_size_sweep", "exp2_block_size_sweep.csv"))]:
        d = pd.read_csv(os.path.join(R, *path))
        if "status" in d:
            d = d[d.status == "ok"]
        wins = d.loc[d.groupby(["seq_len", "batch"]).throughput.idxmax()].block_size.value_counts().sort_index()
        mean = d.groupby("block_size").throughput.mean().round(1)
        won = {int(k): int(v) for k, v in wins.items()}
        avg = {int(k): float(v) for k, v in mean.items()}
        print(f"  {name}: cells won {won} of {sum(won.values())}; mean tok/s by block {avg}")


def table9():
    """Pitfall-2 batch effect vs unpaired-sweep drift, and the interleaved paged/contiguous ratio.

    Batch effect: pre-fix paged tpot_ms at seq 128, batch 16 over batch 1 (exp1 @017b63a).
    Drift: per cell, exp1's unpaired paged/contiguous throughput ratio vs the interleaved median
    (results/paired_layout_check); the largest disagreement factor is the drift we observed.
    """
    pre = pd.read_csv(os.path.join(R, "pre_fix", "exp1_layout_comparison_017b63a.csv"))
    tp = lambda s, b: pre[(pre.cache_type == "paged") & (pre.seq_len == s) & (pre.batch == b)].tpot_ms.iloc[0]
    print("Table 9: pitfall-2 batch effect vs unpaired drift; interleaved paged/contiguous throughput")
    print(f"  pre-fix paged TPOT, seq 128: batch 1 {tp(128, 1):.3f} -> batch 16 {tp(128, 16):.3f} ms "
          f"= {tp(128, 16) / tp(128, 1):.2f}x; batch 1: seq 128 {tp(128, 1):.3f} -> seq 2048 {tp(2048, 1):.3f} ms "
          f"= {tp(2048, 1) / tp(128, 1):.2f}x")
    e1 = pd.read_csv(os.path.join(R, "cache_comparison", "exp1_layout_comparison.csv")).set_index(
        ["cache_type", "seq_len", "batch"])
    paired = json.load(open(os.path.join(R, "paired_layout_check", "paired_layout_check.json")))["cells"]
    drift = []
    for c in paired:
        k = (c["seq_len"], c["batch"])
        unp = e1.throughput[("paged",) + k] / e1.throughput[("contiguous",) + k]
        f = max(unp, c["median_ratio"]) / min(unp, c["median_ratio"])
        drift.append(f)
        slower = sum(p["ratio"] < 1 for p in c["pairs"])
        print(f"  {k[0]}x{k[1]}: interleaved median {c['median_ratio']:.3f} "
              f"(range {c['min_ratio']:.3f}-{c['max_ratio']:.3f}, paged slower in {slower}/{len(c['pairs'])} pairs); "
              f"exp1 unpaired {unp:.3f}; disagreement {f:.2f}x")
    meds = [c["median_ratio"] for c in paired]
    n_pairs = sum(len(c["pairs"]) for c in paired)
    n_slow = sum(p["ratio"] < 1 for c in paired for p in c["pairs"])
    print(f"  interleaved medians {min(meds):.2f}-{max(meds):.2f}; paged slower in {n_slow}/{n_pairs} pairs; "
          f"largest unpaired-vs-interleaved disagreement {max(drift):.2f}x")


def table10():
    """Paged read() gather time vs the paged-contiguous TPOT gap, per cell: ratio of medians with the
    IQRs of gather and gap, and instrumentation overhead as a ratio of median TPOTs. Recomputed from
    the per-rep records with experiments/gather_cost.py's summarize(), so the committed JSONs written
    before that summary existed give the same statistic."""
    sys.path.insert(0, REPO)
    from experiments.gather_cost import summarize
    print("Table 10: paged read() gather vs the paged-contiguous TPOT gap (ms per decode step)")
    for name in ["gather_cost.json", "gather_cost_r15.json"]:
        d = json.load(open(os.path.join(R, "gather_cost", name)))
        print(f"  {name} ({d['reps']} reps):")
        for c in d["cells"]:
            s = summarize(c["reps"])
            flags = [f for f, on in [("SHARE > 1.0", s["share_gt_1"]), ("OVERHEAD > 1.10", s["instrumentation_overhead_ratio"] > 1.10)] if on]
            print(f"    {c['seq_len']}x{c['batch']}: gather {s['median_gather_ms_per_step']:.3f} "
                  f"(IQR {s['gather_iqr_ms'][0]:.3f}-{s['gather_iqr_ms'][1]:.3f}), gap {s['median_gap_ms']:.3f} "
                  f"(IQR {s['gap_iqr_ms'][0]:.3f}-{s['gap_iqr_ms'][1]:.3f}, {'resolved' if s['gap_resolved'] else 'not resolved'}); "
                  f"ratio of medians {s['gather_share_of_gap']:.2f}; overhead {s['instrumentation_overhead_ratio']:.3f}x; "
                  f"share > 1 in {s['reps_with_share_gt_1']}/{len(c['reps'])} reps"
                  + (f"  ** {', '.join(flags)} **" if flags else ""))


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


PAPER_TABLE1_ORDER = ["4L/512d/2048ff", "8L/512d/2048ff", "16L/512d/2048ff", "8L/1024d/2048ff",
                      "8L/512d/4096ff", "12L/768d/3072ff"]
TABLE1_HEADER = ["Architecture", "Cache (KB)", "Activation predicted", "Activation measured",
                 "Predicted total", "Measured total", "Error"]


def table1_cells():
    """Paper Table 1 (per-token memory, predicted vs measured), as the paper prints each cell.

    Same computation as scripts/check_memory_model.py: src.memory_model.predict_slopes on
    results/arch_sweep/arch_sweep.json, integer coefficients, corrected cache term. The cache
    column is the derived value written out exactly (at least two decimals); activations to
    0.01 KB, totals to 0.001 KB, error to 0.01%.
    """
    sys.path.insert(0, REPO)
    from src.memory_model import predict_slopes
    _, rows = predict_slopes(json.load(open(os.path.join(R, "arch_sweep", "arch_sweep.json"))),
                             corrected=True, coeffs="integer")
    by = {r["shape"]: r for r in rows}

    def exact(v):
        s = f"{v:.6f}".rstrip("0")
        return s + "0" * max(0, 2 - len(s.split(".")[1]))

    def err(v):
        return "0.00%" if abs(v) < 0.005 else f"{v:+.2f}%".replace("-", "−")

    out = []
    for shape in PAPER_TABLE1_ORDER:
        r = by[shape]
        n_l, d, ff = (int(x.rstrip("Ldf")) for x in shape.split("/"))
        name = f"{n_l}L / {d} / {ff}"
        if shape == "12L/768d/3072ff":
            name = f"GPT-2 small, {name} (held out)"
        out.append([name, exact(r["cache_kb"]), f"{r['act_pred_kb']:.2f}", f"{r['act_kb']:.2f}",
                    f"{r['predicted']:.3f}", f"{r['measured']:.3f}", err(r["error_pct"])])
    return out


def paper_table(n):
    """Rows of Table n in paper.md (local, gitignored) as printed, or None if paper.md is absent."""
    path = os.path.join(REPO, "paper.md")
    if not os.path.exists(path):
        return None
    lines = open(path, encoding="utf-8").read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith(f"**Table {n}\\.**"))
    rows = []
    for l in lines[start + 1:]:
        if rows and not l.startswith("|"):
            break
        if l.startswith("|") and not l.startswith("| :"):
            rows.append([c.strip().replace("\\+", "+").replace("\\_", "_") for c in l.strip("|").split("|")])
    return rows[1:]                                          # drop the header row


def tex(cell):
    cell = cell.replace("%", r"\%")
    return re.sub(r"^([−+])", lambda m: "$-$" if m.group(1) == "−" else "$+$", cell)


def write_table1():
    """paper/tables/table1.tex (booktabs), and a cell-by-cell check against paper.md's Table 1."""
    cells = table1_cells()
    os.makedirs(TABLES, exist_ok=True)
    body = [" & ".join(tex(c) for c in row) + r" \\" for row in cells]
    src = "\n".join([
        "% Paper Table 1: per-token memory, predicted against measured (KB per token).",
        "% Generated by scripts/make_tables.py from results/arch_sweep/arch_sweep.json. Do not edit.",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        " & ".join(TABLE1_HEADER) + r" \\",
        r"\midrule",
        *body[:-1],
        r"\midrule",
        body[-1],
        r"\bottomrule",
        r"\end{tabular}",
        ""])
    with open(os.path.join(TABLES, "table1.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    print("Paper Table 1 -> paper/tables/table1.tex")
    printed = paper_table(1)
    if printed is None:
        print("  paper.md not present (it is local); cell check skipped")
        return 0
    bad = 0
    for got, want in zip(cells, printed):
        for h, g, w in zip(TABLE1_HEADER, got, want):
            if g != w:
                bad += 1
                print(f"  MISMATCH  {got[0]} / {h}: generated {g!r}, paper.md {w!r}")
    if len(printed) != len(cells):
        bad += 1
        print(f"  MISMATCH  paper.md has {len(printed)} rows, generated {len(cells)}")
    print(f"  {'PASS' if not bad else 'FAIL'}  {len(cells)} rows x {len(TABLE1_HEADER)} columns vs paper.md")
    return bad


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
    table5()
    print()
    table6()
    print()
    table7()
    print()
    table8()
    print()
    table9()
    print()
    table10()
    print()
    failed = check_reported()
    print()
    failed += write_table1()
    for p in args.against:
        failed += compare(p)
    sys.exit(1 if failed else 0)
