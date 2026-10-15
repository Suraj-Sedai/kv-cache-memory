"""Which activation tensors are live at peak, per token?

The memory model's activation term is (3 * d_model + 2 * d_ff) elements per token. This names
the tensors by checking them on the GPU. In eval mode under torch.no_grad, with the same setup
as arch_sweep.py, the forward pass peaks at the GELU in the MLP (src/model_core/mlp.py:22).
At that instant, five per-token tensors are live:

  d_model  block input            made at transformer_block.py:42 (or model.py:26 for layer 0);
                                  held by the loop variable x at model.py:29 until the layer returns
  d_model  post-attention residual  transformer_block.py:36, held as `residual` (:38)
  d_model  LayerNorm-2 output     transformer_block.py:39, held by the block's `x` during :40
  d_ff     fc1 output             mlp.py:21
  d_ff     GELU output            mlp.py:22

The attention phase holds at most 7 * d_model (block input, ln1 output, q, k, v and the paged
read's k and v; attention.py:47-53). That is <= 3 * d_model + 2 * d_ff whenever d_ff >= 2 * d_model,
which holds for every shape swept.

For each shape this wraps GELU, counts the live CUDA storages of size B*T*d_model and B*T*d_ff
at that instant (gc census), and compares memory there with the forward's global peak and with
(3 * d_model + 2 * d_ff) * 2 B * tokens. A warmup forward runs first so one-time allocations
(about 9 MB, token-independent, which belong in the intercept) do not land on the first shape.

Run:  python -m experiments.activation_census
      writes results/activation_census/activation_census.json and its .env.json
"""
import argparse
import gc
import json
import os

import torch

from src.kv_cache.paged import PagedKVCache
from src.model_core.config import ModelConfig
from src.model_core.model import GPTModel
from src.provenance import write_env

SEQ_LEN, BLOCK_SIZE, DECODE_LEN, SEED, BATCH = 1024, 16, 8, 0, 8   # arch_sweep.py's L, P, D
SHAPES = [(8, 512, 8, 2048), (16, 512, 8, 2048), (4, 512, 8, 2048),
          (8, 1024, 16, 2048), (8, 512, 8, 4096), (12, 768, 12, 3072)]
OUT = os.path.join("results", "activation_census", "activation_census.json")
MB = 2 ** 20


def census(n_d, n_ff):
    counts, seen = {"d_model": 0, "d_ff": 0}, set()
    for o in gc.get_objects():
        try:
            if not (isinstance(o, torch.Tensor) and o.is_cuda):
                continue
            s = o.untyped_storage()
            if s.data_ptr() in seen:
                continue
            n = s.nbytes() // o.element_size()
            if n in (n_d, n_ff):
                counts["d_model" if n == n_d else "d_ff"] += 1
                seen.add(s.data_ptr())
        except Exception:
            pass
    return counts


def run_shape(L, d, H, ff, B, seed, probe=True):
    torch.manual_seed(seed)
    total = SEQ_LEN + DECODE_LEN
    cfg = ModelConfig(vocab_size=50257, max_seq_len=total, n_layers=L, n_heads=H, d_model=d, d_ff=ff)
    model = GPTModel(cfg).cuda().half().eval()
    cache = PagedKVCache(n_layers=L, n_heads=H, dim_head=cfg.head_dim, block_size=BLOCK_SIZE,
                         num_blocks=((total + BLOCK_SIZE - 1) // BLOCK_SIZE) * B, batch_size=B,
                         device="cuda", dtype=torch.float16)
    ids = torch.randint(0, cfg.vocab_size, (B, SEQ_LEN), device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    seen = {}
    gelu = torch.nn.functional.gelu

    def gelu_probe(x, *a, **k):
        y = gelu(x, *a, **k)
        if probe and x.size(1) == SEQ_LEN and not seen:          # layer 0 of the prefill
            torch.cuda.synchronize()
            seen["mb"] = (torch.cuda.memory_allocated() - base) / MB
            seen["counts"] = census(B * SEQ_LEN * d, B * SEQ_LEN * ff)
        return y

    for blk in model.layers:
        blk.mlp.activation = gelu_probe
    with torch.no_grad():
        logits = model(ids, kv_cache=cache, use_cache=True, start_pos=0)
        cache.advance(SEQ_LEN)
        for s in range(DECODE_LEN):
            logits = model(logits[:, -1:].argmax(-1), kv_cache=cache, use_cache=True,
                           start_pos=SEQ_LEN + s)
            cache.advance(1)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / MB
    del model, cache, logits
    gc.collect()
    torch.cuda.empty_cache()
    tokens = B * SEQ_LEN
    return dict(shape=f"{L}L/{d}d/{ff}ff", batch=B, tokens=tokens,
                live_d_model_tensors=seen.get("counts", {}).get("d_model"),
                live_d_ff_tensors=seen.get("counts", {}).get("d_ff"),
                mb_at_gelu=round(seen.get("mb", float("nan")), 2),
                forward_peak_mb=round(peak, 2),
                predicted_mb=round((3 * d + 2 * ff) * 2 * tokens / MB, 2))


def main():
    ap = argparse.ArgumentParser(description="Census of per-token activation tensors live at peak.")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output", default=OUT)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs the GPU"

    run_shape(*SHAPES[0], B=args.batch, seed=args.seed, probe=False)     # warmup, discarded
    rows = []
    for L, d, H, ff in SHAPES:
        r = run_shape(L, d, H, ff, B=args.batch, seed=args.seed)
        rows.append(r)
        print(r)
    ok = all(r["live_d_model_tensors"] == 3 and r["live_d_ff_tensors"] == 2 for r in rows)
    print(f"\n3 d_model + 2 d_ff tensors live at the GELU in every shape: {ok}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {args.output}")
    write_env(args.output)


if __name__ == "__main__":
    main()
