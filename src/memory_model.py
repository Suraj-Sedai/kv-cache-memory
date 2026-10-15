"""Per-token memory model: analysis only, no GPU, no measurement.

    bytes per resident token = cache term + activation term

Cache term, uncorrected:  2 * n_layers * d_model * bytes_per_elem
    (K and V, n_heads * head_dim == d_model in ModelConfig).

Cache term, corrected:    uncorrected * pool_correction(seq_len, decode_len, block_size)

WHERE THE CORRECTION COMES FROM (read off the code, not fitted):
experiments/arch_sweep.py sizes the page pool as
    num_blocks = ceil((SEQ_LEN + DECODE_LEN) / BLOCK_SIZE) * batch
and PagedKVCache.__init__ (src/kv_cache/paged.py) allocates that entire pool with
torch.zeros up front. So each sequence holds ceil((1024 + 8) / 16) = 65 pages = 1040
slots for the whole run. But the x-axis of the arch-sweep fit is batch * SEQ_LEN, i.e.
1024 prompt tokens per sequence. Per counted token the cache costs 1040/1024 slots:
    +8 slots  for the DECODE_LEN = 8 decode tokens, and
    +8 slots  of padding that rounds 1032 up to the next 16-slot page boundary.
That is the whole of the extra 16/1024. It is a property of how the harness sizes the
pool and what it counts as a token, not of any extra per-token tensor. The factor is
specific to (seq_len, decode_len, block_size) = (1024, 8, 16). Other runners use other
values (run_precision_sweep.py: DECODE_LEN 60, five seq_lens), so do not reuse 1040/1024 there.
"""
import math

import numpy as np

# arch_sweep.py constants that set the pool size (SEQ_LEN, DECODE_LEN, BLOCK_SIZE).
ARCH_SWEEP_POOL = dict(seq_len=1024, decode_len=8, block_size=16)

# Shapes notebook 05 fits the activation term on; the other three are predictions.
ACT_FIT_SHAPES = ("8L/512d/2048ff", "8L/1024d/2048ff", "8L/512d/4096ff")


def pool_slots_per_seq(seq_len, decode_len, block_size):
    return math.ceil((seq_len + decode_len) / block_size) * block_size


def pool_correction(seq_len, decode_len, block_size):
    """Allocated cache slots per counted (prompt) token. 1040/1024 for arch_sweep."""
    return pool_slots_per_seq(seq_len, decode_len, block_size) / seq_len


def cache_kb_per_token(n_layers, d_model, bytes_per_elem=2.0, correction=1.0):
    return 2 * n_layers * d_model * bytes_per_elem / 1024 * correction


def shape_name(r):
    return f"{r['n_layers']}L/{r['d_model']}d/{r['d_ff']}ff"


def predict_slopes(arch_rows, correction, fit_on=ACT_FIT_SHAPES):
    """Notebook 05's model with a given cache correction.

    activation KB/token = c0 + c1 * d_model + c2 * d_ff, solved exactly on `fit_on`;
    predicted slope = cache term + activation term. Returns (coeffs, rows) where each
    row carries shape, measured slope, cache, activation residual, prediction, error %.
    """
    rows = []
    for r in arch_rows:
        cache = cache_kb_per_token(r["n_layers"], r["d_model"], correction=correction)
        rows.append(dict(shape=shape_name(r), n_layers=r["n_layers"], d_model=r["d_model"],
                         d_ff=r["d_ff"], measured=r["slope_kb_per_token"], cache_kb=cache,
                         act_kb=r["slope_kb_per_token"] - cache))
    fit = [r for r in rows if r["shape"] in fit_on]
    A = np.array([[1.0, r["d_model"], r["d_ff"]] for r in fit])
    coeffs = np.linalg.solve(A, np.array([r["act_kb"] for r in fit]))
    for r in rows:
        r["predicted"] = r["cache_kb"] + coeffs[0] + coeffs[1] * r["d_model"] + coeffs[2] * r["d_ff"]
        r["error_pct"] = 100 * (r["predicted"] - r["measured"]) / r["measured"]
        r["held_out"] = r["shape"] not in fit_on
    return coeffs, rows
