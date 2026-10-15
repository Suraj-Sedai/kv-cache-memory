"""Per-token memory model: analysis only, no GPU, no measurement.

    KB per counted token = cache term + activation term

CACHE TERM. The paged pool is allocated in whole pages, up front, for every sequence:
    allocated slots per sequence = ceil((L + D) / P) * P
with L = prompt length, D = decode length, P = page (block) size. experiments/arch_sweep.py
sizes the pool exactly this way (num_blocks = ceil((SEQ_LEN + DECODE_LEN) / BLOCK_SIZE) * batch)
and PagedKVCache.__init__ (src/kv_cache/paged.py) allocates all of it with torch.zeros. The
fit's x-axis counts batch * L prompt tokens, so per counted token the cache costs

    2 * n_layers * d_model * bytes_per_elem * (allocated slots / L)

The ratio exceeds 1 for two reasons, both read off the code: the D decode tokens, and padding
up to the next page boundary. For the arch sweep, (L, D, P) = (1024, 8, 16) gives
ceil(1032 / 16) * 16 = 1040 slots: 8 decode slots plus 8 padding slots. It is a property of
how the harness sizes the pool and what it counts as a token, not an extra per-token tensor.
Other runners use other (L, D) (run_precision_sweep.py: D = 60, five values of L), so the
ratio must be recomputed per configuration, never copied.

ACTIVATION TERM. activation KB/token = (a * d_model + b * d_ff) * bytes_per_elem / 1024 (+ c0).
Two coefficient sets are available:
  "fitted"   a, b, c0 solved exactly on ACT_FIT_SHAPES (a = 3.030, b = 1.997, c0 = -0.018 KB)
  "integer"  a = 3, b = 2, c0 = 0 (the paper's Table 1 form)
"""
import ast
import math
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH_SWEEP_SCRIPT = os.path.join(REPO, "experiments", "arch_sweep.py")

# Shapes the fitted activation coefficients are solved on; the other three are predictions.
ACT_FIT_SHAPES = ("8L/512d/2048ff", "8L/1024d/2048ff", "8L/512d/4096ff")
INTEGER_COEFFS = (0.0, 3.0, 2.0)          # (c0 KB, a, b)
BYTES_FP16 = 2.0


def arch_sweep_config(path=ARCH_SWEEP_SCRIPT):
    """(L, D, P) = (SEQ_LEN, DECODE_LEN, BLOCK_SIZE) as assigned in experiments/arch_sweep.py.
    Parsed, not imported, so analysis needs no torch."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            names = [t.id for t in node.targets[0].elts]
            if "SEQ_LEN" in names:
                vals = dict(zip(names, ast.literal_eval(node.value)))
                return vals["SEQ_LEN"], vals["DECODE_LEN"], vals["BLOCK_SIZE"]
    raise ValueError(f"SEQ_LEN/DECODE_LEN/BLOCK_SIZE not found in {path}")


def allocated_slots(L, D, P):
    """Cache slots one sequence holds: ceil((L + D) / P) * P."""
    return math.ceil((L + D) / P) * P


def cache_kb_per_token(n_layers, d_model, slots_per_seq, counted_per_seq, bytes_per_elem=BYTES_FP16):
    """Cache KB per counted token. slots_per_seq == counted_per_seq gives the uncorrected term."""
    return 2 * n_layers * d_model * bytes_per_elem / 1024 * slots_per_seq / counted_per_seq


def shape_name(r):
    return f"{r['n_layers']}L/{r['d_model']}d/{r['d_ff']}ff"


def predict_slopes(arch_rows, corrected=True, coeffs="fitted", config=None):
    """Predicted per-token slope for each arch_sweep row.

    corrected: cache term over allocated slots (True) or over counted tokens only (False).
    coeffs:    "fitted" (solved on ACT_FIT_SHAPES) or "integer" (3 * d_model + 2 * d_ff).
    Returns ((c0, a, b), rows); a and b are in tensors (elements per token), c0 in KB.
    """
    L, D, P = config or arch_sweep_config()
    slots = allocated_slots(L, D, P) if corrected else L
    rows = []
    for r in arch_rows:
        cache = cache_kb_per_token(r["n_layers"], r["d_model"], slots, L)
        rows.append(dict(shape=shape_name(r), n_layers=r["n_layers"], d_model=r["d_model"],
                         d_ff=r["d_ff"], measured=r["slope_kb_per_token"], cache_kb=cache,
                         act_kb=r["slope_kb_per_token"] - cache))
    kb = BYTES_FP16 / 1024                       # KB per element of width 1
    if coeffs == "integer":
        c0, a, b = INTEGER_COEFFS
    elif coeffs == "fitted":
        fit = [r for r in rows if r["shape"] in ACT_FIT_SHAPES]
        A = np.array([[1.0, r["d_model"] * kb, r["d_ff"] * kb] for r in fit])
        c0, a, b = np.linalg.solve(A, np.array([r["act_kb"] for r in fit]))
    else:
        raise ValueError(coeffs)
    for r in rows:
        r["act_pred_kb"] = c0 + (a * r["d_model"] + b * r["d_ff"]) * kb
        r["predicted"] = r["cache_kb"] + r["act_pred_kb"]
        r["error_pct"] = 100 * (r["predicted"] - r["measured"]) / r["measured"]
        r["held_out"] = r["shape"] not in ACT_FIT_SHAPES
    return (c0, a, b), rows
