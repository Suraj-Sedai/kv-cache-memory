"""Achievable FP16 matmul throughput on this GPU (TFLOP/s).

Needed for the arithmetic share of a decode step (scripts/make_tables.py Table 5): decode-step
FLOPs divided by this throughput, divided by the measured TPOT. Square FP16 matmuls at n = 4096
and 8192, --warmup untimed iterations, then the median of --iters timed iterations (CUDA events).
FLOPs per matmul = 2 * n^3.

Run:  python -m experiments.matmul_peak
      writes results/matmul_peak/matmul_peak.json and its .env.json
"""
import argparse
import json
import os
import statistics

import torch

from src.provenance import write_env

SIZES = [4096, 8192]
OUT = os.path.join("results", "matmul_peak", "matmul_peak.json")


def measure(n, warmup, iters):
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    for _ in range(warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    ms = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        ms.append(start.elapsed_time(end))
    med = statistics.median(ms)
    return dict(n=n, flops=2 * n ** 3, median_ms=round(med, 4), min_ms=round(min(ms), 4),
                max_ms=round(max(ms), 4), tflops=round(2 * n ** 3 / (med * 1e-3) / 1e12, 2))


def main():
    ints = lambda s: [int(x) for x in s.split(",")]
    ap = argparse.ArgumentParser(description="Achievable FP16 matmul TFLOP/s.")
    ap.add_argument("--sizes", type=ints, default=SIZES, help="comma-separated n (default 4096,8192)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--output", default=OUT)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs the GPU"

    rows = []
    for n in args.sizes:
        rows.append(measure(n, args.warmup, args.iters))
        print(rows[-1])
    best = max(rows, key=lambda r: r["tflops"])
    out = dict(dtype="float16", warmup=args.warmup, iters=args.iters, sizes=rows,
               achievable_tflops=best["tflops"], achievable_from_n=best["n"])
    print(f"\nachievable FP16 matmul: {best['tflops']} TFLOP/s (n = {best['n']})")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}")
    write_env(args.output)


if __name__ == "__main__":
    main()
