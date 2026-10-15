"""Experiment 2 — paged KV cache block-size sweep.

Question: how does block_size trade fragmentation against per-token lookup
overhead? Smaller blocks -> finer packing (lower fragmentation) but more pages
to gather per read; larger blocks -> cheaper gather but coarser last-page waste.

seq_lens are deliberately NON-multiples of the block sizes so fragmentation
actually varies (powers of two give zero waste -> a flat, useless Figure 2).

NOTE: decode on this GPU is launch-bound (throughput CV up to 42.59% in exp1,
46.79% in exp3), so the fragmentation/memory columns are the clean signal;
TPOT-vs-block_size may stay within the noise. A flat TPOT curve here is the known
launch-bound regime, not a new bug.

PORTED to the batched cache interface (fceab20). The committed CSV
(results/block_size_sweep/exp2_block_size_sweep.csv) was rerun after that port, the SDPA
switch and the lm_head slice (committed in ae0cb58). frag_ratio is waste over TOTAL
allocation, not waste over the last block; the pre-port files in results/pre_fix/ use the
old definition and are not comparable.
"""
import argparse
import csv
import os
import statistics
import time

import torch

from src.inference.controller import InferenceController
from src.kv_cache.paged import PagedKVCache
from src.model_core.config import ModelConfig
from src.model_core.model import GPTModel

N_LAYERS, N_HEADS, D_MODEL, D_FF = 8, 8, 512, 2048
SEED = 0   # added in the cleanup; the committed CSV was produced unseeded

BLOCK_SIZES = [8, 16, 32, 64]
SEQ_LENS    = [100, 300, 600, 1100, 2000]   # non-multiples -> frag varies
BATCH_SIZES = [1, 8, 16]
DECODE_LEN  = 60
N_WARMUP    = 3
N_MEASURE   = 20
BUDGET_S    = 20.0   # one warmup slower than this -> cell thrashes -> 'slow'

OUTPUT_PATH = os.path.join("results", "block_size_sweep", "exp2_block_size_sweep.csv")
FIELDNAMES  = ["cache_type", "seq_len", "batch", "block_size", "status",
               "ttft_ms", "tpot_ms", "throughput", "throughput_cv",
               "peak_memory_mb", "cache_memory_mb", "frag_ratio"]


def make_paged(model_config, batch, max_seq_len, block_size, device, dtype):
    per_seq_blocks = (max_seq_len + block_size - 1) // block_size
    return PagedKVCache(
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        dim_head=model_config.head_dim,
        block_size=block_size,
        num_blocks=per_seq_blocks * batch,   # shared pool across all seqs
        batch_size=batch,
        device=device,
        dtype=dtype,
    )


def measure_cache_memory_frag(model_config, batch, max_seq_len, block_size, final_len, device, dtype):
    """Fill a throwaway paged cache to the run's final state and read its OWN
    memory_bytes()/fragmentation_ratio() — exact and dtype-correct. Done
    out-of-band because controller.generate() frees the cache at cleanup.

    Batched API: one write for the whole batch. Writing layer 0 only is enough —
    a page spans all layers, so memory_bytes() already covers the full pool draw.
    """
    cache = make_paged(model_config, batch, max_seq_len, block_size, device, dtype)
    dummy = torch.zeros(batch, model_config.n_heads, final_len, model_config.head_dim,
                        device=device, dtype=dtype)
    cache.write(0, dummy, dummy)
    cache.advance(final_len)
    mem_mb = cache.memory_bytes() / (1024 * 1024)
    frag = cache.fragmentation_ratio()
    cache.free()
    del cache, dummy
    if device == "cuda":
        torch.cuda.empty_cache()
    return round(mem_mb, 2), round(frag, 4)


def run_one(model, model_config, cache, prompt_ids):
    controller = InferenceController(model, model_config)
    return controller.generate(prompt_ids, max_new_tokens=DECODE_LEN, cache=cache, use_cache=True)


def parse_args():
    """Defaults are the constants above; parsing happens before any GPU work."""
    ints = lambda s: [int(x) for x in s.split(",")]
    ap = argparse.ArgumentParser(description="Exp 2: paged KV cache block-size sweep.")
    ap.add_argument("--block-sizes", type=ints, default=BLOCK_SIZES, help="comma-separated (default: %(default)s)")
    ap.add_argument("--seq-lens", type=ints, default=SEQ_LENS, help="comma-separated (default: %(default)s)")
    ap.add_argument("--batches", type=ints, default=BATCH_SIZES, help="comma-separated (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output", default=OUTPUT_PATH, help="CSV path (overwritten)")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}  |  model: {N_LAYERS}L/{N_HEADS}H/{D_MODEL}d  |  FP16, paged only")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = []

    for block_size in args.block_sizes:
        for seq_len in args.seq_lens:
            for batch in args.batches:
                max_seq_len = seq_len + DECODE_LEN
                final_len = seq_len + DECODE_LEN
                model_config = ModelConfig(max_seq_len=max_seq_len, n_layers=N_LAYERS,
                                           n_heads=N_HEADS, d_model=D_MODEL, d_ff=D_FF)
                torch.manual_seed(args.seed)   # identical weights + prompts per cell
                model = GPTModel(model_config).to(device).half()   # FP16
                io_dtype = next(model.parameters()).dtype
                prompt_ids = torch.randint(0, model_config.vocab_size, (batch, seq_len), device=device)

                base = {"cache_type": "paged", "seq_len": seq_len, "batch": batch,
                        "block_size": block_size}
                blank_timing = {"ttft_ms": "", "tpot_ms": "", "throughput": "",
                                "throughput_cv": "", "peak_memory_mb": ""}
                try:
                    cache_mem_mb, frag = measure_cache_memory_frag(
                        model_config, batch, max_seq_len, block_size, final_len, device, io_dtype)
                    base.update(cache_memory_mb=cache_mem_mb, frag_ratio=frag)

                    t0 = time.perf_counter()
                    run_one(model, model_config,
                            make_paged(model_config, batch, max_seq_len, block_size, device, io_dtype),
                            prompt_ids)
                    if time.perf_counter() - t0 > BUDGET_S:
                        row = {**base, "status": "slow", **blank_timing}
                    else:
                        for _ in range(N_WARMUP - 1):
                            run_one(model, model_config,
                                    make_paged(model_config, batch, max_seq_len, block_size, device, io_dtype),
                                    prompt_ids)
                        metrics_list = []
                        for _ in range(N_MEASURE):
                            cache = make_paged(model_config, batch, max_seq_len, block_size, device, io_dtype)
                            metrics_list.append(run_one(model, model_config, cache, prompt_ids).metrics)

                        throughputs = [m.throughput_tokens_per_sec for m in metrics_list]
                        thr_median = statistics.median(throughputs)
                        thr_cv = (statistics.pstdev(throughputs) / thr_median * 100) if thr_median else 0.0
                        row = {
                            **base, "status": "ok",
                            "ttft_ms":    round(statistics.median(m.ttft_ms for m in metrics_list), 3),
                            "tpot_ms":    round(statistics.median(m.tpot_p50_ms for m in metrics_list), 3),
                            "throughput": round(thr_median, 1),
                            "throughput_cv": round(thr_cv, 2),
                            "peak_memory_mb": round(statistics.median(m.peak_memory_mb for m in metrics_list), 2),
                        }
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    row = {**base, "status": "oom", **blank_timing,
                           "cache_memory_mb": base.get("cache_memory_mb", ""),
                           "frag_ratio": base.get("frag_ratio", "")}

                results.append(row)
                print(row)
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
