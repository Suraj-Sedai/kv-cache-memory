"""Pitfall 4 evidence: with no software memory cap, does the Windows (WDDM) driver
spill to host RAM instead of raising OOM?

Runs the two frontier cells from oom_frontier.py (8192x64 and 16384x32, FP16 and INT8)
WITHOUT set_per_process_memory_fraction, and records peak torch.cuda.memory_allocated
and memory_reserved against the device's physical memory. If the sysmem fallback is
on, FP16 completes with allocated > physical (the first pass of oom_frontier.py saw
14009 MB on an 11.9 GB card). If it is off, the cell raises OOM, and that is recorded.

The cell code is oom_frontier.py's try_cell, unchanged except that it also reads
reserved memory and timing. Expect the spilled cells to be slow (PCIe-backed memory).

Run:  python -m experiments.wddm_spill          (writes results/pre_fix/wddm_spill.json)
"""
import argparse
import json
import os
import platform
import subprocess
import time

import torch

from experiments.oom_frontier import (BLOCK_SIZE, D_FF, D_MODEL, DECODE_LEN, N_HEADS,
                                      N_LAYERS, cleanup, is_oom)
from src.kv_cache.int8_paged import INT8KVCache
from src.kv_cache.paged import PagedKVCache
from src.model_core.config import ModelConfig
from src.model_core.model import GPTModel

CELLS = [(8192, 64), (16384, 32)]
SEED = 0
OUT = os.path.join("results", "pre_fix", "wddm_spill.json")
MB = 1024 ** 2


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def git_commit():
    head = sh(["git", "rev-parse", "--short", "HEAD"])
    dirty = sh(["git", "status", "--porcelain"])
    return head + ("-dirty" if dirty and not dirty.startswith("unavailable") else "")


def run_cell(seq_len, batch, quantized, seed):
    cleanup()
    total = seq_len + DECODE_LEN
    per_seq = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    cfg = ModelConfig(vocab_size=50257, max_seq_len=total, n_layers=N_LAYERS,
                      n_heads=N_HEADS, d_model=D_MODEL, d_ff=D_FF)
    model = cache = None
    t0 = time.perf_counter()
    try:
        torch.manual_seed(seed)
        model = GPTModel(cfg).to("cuda").half().eval()
        kw = dict(n_layers=N_LAYERS, n_heads=N_HEADS, dim_head=cfg.head_dim,
                  block_size=BLOCK_SIZE, num_blocks=per_seq * batch,
                  batch_size=batch, device="cuda")
        cache = (INT8KVCache(**kw, compute_dtype=torch.float16) if quantized
                 else PagedKVCache(**kw, dtype=torch.float16))
        ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), device="cuda")
        with torch.no_grad():
            logits = model(ids, kv_cache=cache, use_cache=True, start_pos=0)
            cache.advance(seq_len)
            for s in range(DECODE_LEN):
                nxt = logits[:, -1:].argmax(-1)
                logits = model(nxt, kv_cache=cache, use_cache=True, start_pos=seq_len + s)
                cache.advance(1)
        torch.cuda.synchronize()
        return dict(status="ok",
                    peak_allocated_mb=round(torch.cuda.max_memory_allocated() / MB, 1),
                    peak_reserved_mb=round(torch.cuda.max_memory_reserved() / MB, 1),
                    cache_mb=round(cache.memory_bytes() / MB, 1),
                    wall_s=round(time.perf_counter() - t0, 2))
    except Exception as e:
        if not is_oom(e):
            raise
        return dict(status="oom", error=str(e).splitlines()[0][:300],
                    peak_allocated_mb=round(torch.cuda.max_memory_allocated() / MB, 1),
                    peak_reserved_mb=round(torch.cuda.max_memory_reserved() / MB, 1),
                    wall_s=round(time.perf_counter() - t0, 2))
    finally:
        del model, cache
        cleanup()


def main():
    ap = argparse.ArgumentParser(description="Record WDDM sysmem spill with no memory cap.")
    ap.add_argument("--cells", type=lambda s: [tuple(int(v) for v in c.split("x")) for c in s.split(",")],
                    default=CELLS, help="comma-separated SEQxBATCH (default: 8192x64,16384x32)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output", default=OUT)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs the GPU"
    torch.cuda.set_per_process_memory_fraction(1.0)   # explicitly: no software cap
    props = torch.cuda.get_device_properties(0)
    physical_mb = props.total_memory / MB
    meta = dict(
        gpu=torch.cuda.get_device_name(0),
        driver_version=sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        driver_model=sh(["nvidia-smi", "--query-gpu=driver_model.current", "--format=csv,noheader"]),
        physical_memory_mb=round(physical_mb, 1),
        torch=torch.__version__, cuda=torch.version.cuda, os=platform.platform(),
        commit=git_commit(), software_cap="none (set_per_process_memory_fraction(1.0))",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    print(json.dumps(meta, indent=2))

    rows = []
    for seq_len, batch in args.cells:
        for label, q in (("fp16", False), ("int8", True)):
            r = dict(seq_len=seq_len, batch=batch, dtype=label, **run_cell(seq_len, batch, q, args.seed))
            r["allocated_exceeds_physical"] = r["peak_allocated_mb"] > physical_mb
            rows.append(r)
            print(r)

    spilled = [r for r in rows if r["status"] == "ok" and r["allocated_exceeds_physical"]]
    meta["verdict"] = (f"{len(spilled)} cell(s) completed with peak allocated above physical "
                       f"memory ({physical_mb:.0f} MB): the driver spilled instead of raising OOM"
                       if spilled else
                       "no cell completed above physical memory: no spill observed on this run")
    print("\n" + meta["verdict"])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dict(meta=meta, cells=rows), f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
