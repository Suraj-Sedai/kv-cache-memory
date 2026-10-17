"""Paired paged/contiguous throughput: is the exp1 ratio real, or drift between two sweeps?

run_cache_comparison.py loops cache_type outermost (run_cache_comparison.py:165), so every
contiguous cell finishes before any paged cell starts; each exp1 "pair" is separated by a whole
sweep with unlocked clocks. This runs both layouts interleaved, A B A B ... (A = contiguous,
B = paged), --reps times in one process, per cell, with the same model, weights and prompt, and
reports the paged/contiguous throughput ratio for each adjacent A,B pair: median and spread.

Each A or B block is --runs-per-block timed generations (median taken), after --warmup untimed
generations of each layout. The generation path is run_cache_comparison.py's make_cache and
run_one, unchanged.

Run:  python -m experiments.paired_layout_check
      writes results/paired_layout_check/paired_layout_check.json and its .env.json
"""
import argparse
import json
import os
import statistics

import torch

from experiments.run_cache_comparison import (D_FF, D_MODEL, DECODE_LEN, N_HEADS, N_LAYERS,
                                              SEED, make_cache, run_one)
from src.model_core.config import ModelConfig
from src.model_core.model import GPTModel
from src.provenance import write_env

CELLS = [(512, 1), (2048, 1), (2048, 16)]
OUT = os.path.join("results", "paired_layout_check", "paired_layout_check.json")
ORDER = ("contiguous", "paged")          # A, B


def block(model, mc, layout, batch, prompt, n):
    """Median throughput (tok/s) over n timed generations of one layout."""
    io_dtype = next(model.parameters()).dtype
    tps = []
    for _ in range(n):
        cache = make_cache(layout, mc, batch, prompt.size(1) + DECODE_LEN, "cuda", io_dtype)
        tps.append(run_one(model, mc, cache, prompt).metrics.throughput_tokens_per_sec)
    return statistics.median(tps)


def main():
    cells = lambda s: [tuple(int(v) for v in c.split("x")) for c in s.split(",")]
    ap = argparse.ArgumentParser(description="Interleaved (ABAB) paged vs contiguous throughput.")
    ap.add_argument("--cells", type=cells, default=CELLS, help="SEQxBATCH list (default 512x1,2048x1,2048x16)")
    ap.add_argument("--reps", type=int, default=5, help="number of A,B pairs per cell")
    ap.add_argument("--runs-per-block", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3, help="untimed generations of each layout first")
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
        for layout in ORDER:
            block(model, mc, layout, batch, prompt, args.warmup)
        pairs = []
        for rep in range(args.reps):
            tp = {layout: block(model, mc, layout, batch, prompt, args.runs_per_block) for layout in ORDER}
            pairs.append(dict(rep=rep + 1, contiguous_tps=round(tp["contiguous"], 1),
                              paged_tps=round(tp["paged"], 1),
                              ratio=round(tp["paged"] / tp["contiguous"], 4)))
            print(f"{seq_len}x{batch} rep {rep + 1}: {pairs[-1]}")
        r = [p["ratio"] for p in pairs]
        row = dict(seq_len=seq_len, batch=batch, order="ABAB (A=contiguous, B=paged)", pairs=pairs,
                   median_ratio=round(statistics.median(r), 4), min_ratio=min(r), max_ratio=max(r),
                   spread_max_over_min=round(max(r) / min(r), 3))
        rows.append(row)
        print(f"{seq_len}x{batch}: paged/contiguous median {row['median_ratio']} "
              f"(range {row['min_ratio']}-{row['max_ratio']})\n")
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dict(reps=args.reps, runs_per_block=args.runs_per_block, warmup=args.warmup,
                       seed=args.seed, decode_len=DECODE_LEN, cells=rows), f, indent=2)
    print(f"wrote {args.output}")
    write_env(args.output)


if __name__ == "__main__":
    main()
