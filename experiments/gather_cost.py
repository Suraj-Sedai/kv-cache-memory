"""How much of the paged-vs-contiguous decode gap is the paged read() gather?

PagedKVCache.read (src/kv_cache/paged.py:69-74) copies a sequence's pages into a dense
[B, H, n, Hd] tensor before standard attention. A fused paged kernel would not make that copy.
This measures, per decode step, the time spent inside read() (the gather) and compares it with
the per-token latency gap between the two layouts:

    share = gather_ms_per_step / (TPOT_paged - TPOT_contiguous)

Per cell, in one process, with one model, weights and prompt, each rep runs in order:
  A  contiguous generation -> TPOT (tpot_p50_ms)
  B  paged generation, uninstrumented -> TPOT
  C  paged generation with CUDA events around every read() -> gather ms per decode step
     (sum over layers, median over the decode steps), plus that run's own TPOT
A and B are the same generation path as run_cache_comparison.py (make_cache, run_one). Timing
is kept off B so the instrumentation does not inflate TPOT_paged. The share uses A and B's TPOTs;
C's TPOT is reported only to show the instrumentation overhead (tpot_paged_instrumented_ms -
tpot_paged_ms). A share above 1.0 (gather longer than the whole gap) is flagged per rep and per
cell: it means the events overstate the gather or the gap is noise-sized.

The events bracket the whole read() call on the stream, so the gather time includes the index
arithmetic (paged.py _index) and any launch gaps between those kernels, not only the copy.
Only decode-step reads are counted (cache.seq_len > 0; the prefill read happens before
advance()).

Run:  python -m experiments.gather_cost
      writes results/gather_cost/gather_cost.json and its .env.json
"""
import argparse
import json
import os
import statistics
from collections import defaultdict

import torch

from experiments.run_cache_comparison import (D_FF, D_MODEL, DECODE_LEN, N_HEADS, N_LAYERS,
                                              SEED, make_cache, run_one)
from src.model_core.config import ModelConfig
from src.model_core.model import GPTModel
from src.provenance import write_env

CELLS = [(512, 1), (2048, 1), (2048, 16)]
OUT = os.path.join("results", "gather_cost", "gather_cost.json")


def tpot(model, mc, layout, prompt):
    cache = make_cache(layout, mc, prompt.size(0), prompt.size(1) + DECODE_LEN, "cuda",
                       next(model.parameters()).dtype)
    return run_one(model, mc, cache, prompt).metrics.tpot_p50_ms


def gather_per_step(model, mc, prompt):
    """Median over decode steps of the summed read() time across layers (ms), the step count,
    and this instrumented run's TPOT."""
    cache = make_cache("paged", mc, prompt.size(0), prompt.size(1) + DECODE_LEN, "cuda",
                       next(model.parameters()).dtype)
    read, events = cache.read, []

    def timed_read(layer_idx):
        if cache.seq_len == 0:                     # prefill: not a decode step
            return read(layer_idx)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = read(layer_idx)
        end.record()
        events.append((cache.seq_len, start, end))
        return out

    cache.read = timed_read
    t = run_one(model, mc, cache, prompt).metrics.tpot_p50_ms
    torch.cuda.synchronize()
    per_step = defaultdict(float)
    for step, start, end in events:
        per_step[step] += start.elapsed_time(end)
    return statistics.median(per_step.values()), len(per_step), t


def main():
    cells = lambda s: [tuple(int(v) for v in c.split("x")) for c in s.split(",")]
    ap = argparse.ArgumentParser(description="Paged read() gather time vs the paged-contiguous TPOT gap.")
    ap.add_argument("--cells", type=cells, default=CELLS, help="SEQxBATCH list (default 512x1,2048x1,2048x16)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3, help="untimed generations of each kind first")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output", default=OUT)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs the GPU"

    rows = []
    for seq_len, batch in args.cells:
        mc = ModelConfig(max_seq_len=seq_len + DECODE_LEN, n_layers=N_LAYERS, n_heads=N_HEADS,
                         d_model=D_MODEL, d_ff=D_FF)
        torch.manual_seed(args.seed)
        model = GPTModel(mc).to("cuda").half()
        prompt = torch.randint(0, mc.vocab_size, (batch, seq_len), device="cuda")
        for _ in range(args.warmup):
            tpot(model, mc, "contiguous", prompt)
            tpot(model, mc, "paged", prompt)
            gather_per_step(model, mc, prompt)
        reps = []
        for rep in range(args.reps):
            t_c = tpot(model, mc, "contiguous", prompt)
            t_p = tpot(model, mc, "paged", prompt)
            g, n_steps, t_pi = gather_per_step(model, mc, prompt)
            gap = t_p - t_c
            share = g / gap if gap > 0 else None
            reps.append(dict(rep=rep + 1, tpot_contiguous_ms=round(t_c, 4), tpot_paged_ms=round(t_p, 4),
                             tpot_paged_instrumented_ms=round(t_pi, 4),
                             instrumentation_overhead_ms=round(t_pi - t_p, 4),
                             gap_ms=round(gap, 4), gather_ms_per_step=round(g, 4), decode_steps=n_steps,
                             gather_share_of_gap=round(share, 4) if share is not None else None,
                             share_gt_1=share is not None and share > 1.0))
            print(f"{seq_len}x{batch} rep {rep + 1}: {reps[-1]}"
                  + ("  ** SHARE > 1.0 **" if reps[-1]["share_gt_1"] else ""))
        shares = [r["gather_share_of_gap"] for r in reps if r["gather_share_of_gap"] is not None]
        med = lambda k: round(statistics.median(r[k] for r in reps), 4)
        row = dict(seq_len=seq_len, batch=batch, reps=reps,
                   median_tpot_contiguous_ms=med("tpot_contiguous_ms"), median_tpot_paged_ms=med("tpot_paged_ms"),
                   median_tpot_paged_instrumented_ms=med("tpot_paged_instrumented_ms"),
                   median_instrumentation_overhead_ms=med("instrumentation_overhead_ms"),
                   median_gap_ms=med("gap_ms"), median_gather_ms_per_step=med("gather_ms_per_step"),
                   median_gather_share_of_gap=round(statistics.median(shares), 4) if shares else None,
                   min_share=min(shares) if shares else None, max_share=max(shares) if shares else None,
                   reps_with_nonpositive_gap=sum(r["gather_share_of_gap"] is None for r in reps),
                   reps_with_share_gt_1=sum(r["share_gt_1"] for r in reps))
        row["share_gt_1"] = row["median_gather_share_of_gap"] is not None and row["median_gather_share_of_gap"] > 1.0
        rows.append(row)
        print(f"{seq_len}x{batch}: gather {row['median_gather_ms_per_step']} ms/step, gap "
              f"{row['median_gap_ms']} ms -> share {row['median_gather_share_of_gap']} "
              f"(range {row['min_share']}-{row['max_share']}); TPOT paged {row['median_tpot_paged_ms']} ms, "
              f"instrumented {row['median_tpot_paged_instrumented_ms']} ms"
              + (f"\n  ** SHARE > 1.0: median {row['median_gather_share_of_gap']}, "
                 f"{row['reps_with_share_gt_1']}/{len(reps)} reps **" if row["reps_with_share_gt_1"] else "") + "\n")
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dict(order="per rep: contiguous TPOT, paged TPOT, paged instrumented",
                       share="gather_ms_per_step (instrumented run) / (tpot_paged - tpot_contiguous) (uninstrumented runs)",
                       flag="share_gt_1 marks share > 1.0",
                       cells_with_share_gt_1=[f"{r['seq_len']}x{r['batch']}" for r in rows if r["reps_with_share_gt_1"]],
                       reps=args.reps, warmup=args.warmup, seed=args.seed, decode_len=DECODE_LEN,
                       cells=rows), f, indent=2)
    print(f"wrote {args.output}")
    write_env(args.output)


if __name__ == "__main__":
    main()
