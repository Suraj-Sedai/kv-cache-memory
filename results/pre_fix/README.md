# Pre-fix outputs

These files were produced **before** one or more of the four measurement fixes. They are kept as
evidence for the pitfalls section and must not be used as sources for current results. Every file
except `wddm_spill.json` was extracted byte-for-byte from git history (`git show <commit>:<path>`);
the blob hashes match.

The four pitfalls:

1. **LM head** — the LM head ran over every prefill position instead of the last one. Fixed in
   237ad2d (twin 98d0e75).
2. **Per-sequence loop** — attention and cache ops looped over the batch in Python. Fixed in
   fceab20 (twin 90ed952), "Batch KV cache interface".
3. **Oracle reservation** — the contiguous cache reserved exactly the final length, so its waste
   is 0 by construction. *Withdrawn, not fixed:* the current `run_cache_comparison.py` still does
   this and still reports 0.0 (see its `make_cache` docstring).
4. **WDDM spill** — the Windows driver spilled to host RAM instead of raising OOM. Worked around
   in e414dee by enforcing budgets with `torch.cuda.set_per_process_memory_fraction`.

| File | Source commit | Pitfall | Exact evidence |
|---|---|---|---|
| `exp3_precision_sweep_e383030.csv` | e383030 (twin c82b120), `experiments/results/exp3_precision_sweep.csv` | 1. LM head | `paged_fp16` rows: `cache_memory_mb / peak_memory_mb` is 4.1% to 13.8% across the 16 `ok` cells (min: seq 512, batch 1, 9.0 / 217.01; max: seq 512, batch 32, 288.0 / 2081.32). The post-fix file gives 5.2% to 58.0% over 20 cells, with the same `cache_memory_mb` (9.0 at 512×1). Also, `status` is `slow`/`oom` at 4096×32, 8192×8, 8192×16 and 8192×32 for both dtypes; all 40 cells are `ok` post-fix. |
| `exp1_layout_comparison_e383030.csv` | e383030 (twin c82b120), `experiments/results/exp1_layout_comparison.csv` | 2. Per-sequence loop (also 1 and 3) | **Pitfall 2:** paged `tpot_ms` grows with batch and stays flat across seq_len: 4.920 → 23.228 ms from batch 1 to 16 at seq 128; 4.920 → 5.093 ms from seq 128 to 2048 at batch 1 (full row at seq 128: 4.92 / 8.04 / 13.18 / 23.23 for batch 1 / 4 / 8 / 16). Post-fix paged is 10.62 to 11.21 at seq 128 for every batch. **Pitfall 1:** `peak_memory_mb` is 332.95 (contiguous, 128×1) against 161.27 post-fix, with `cache_memory_mb` 5.88. **Pitfall 3:** contiguous `frag_ratio` is 0.0 in all 19 `ok` rows. |
| `exp1_layout_comparison_0584aa1.csv` | 0584aa1, `experiments/results/exp1_layout_comparison2.csv` (old 9-column schema) | 3. Oracle reservation | Contiguous `frag_ratio` is exactly 0.0 in all 20 contiguous rows; paged is 0.25 in every row. This is the old per-last-block definition of `frag_ratio`, not comparable to current values. |
| `exp2_block_size_sweep_13fbfae.csv` | 13fbfae, `experiments/results/exp2_block_size_sweep.csv` | None of the four. Evidence for the withdrawn block-size claim (noise). | Row `paged,1100,16,32`: `throughput` 226.6, `throughput_cv` 5.52, `tpot_ms` 62.747. |
| `exp2_block_size_sweep_e383030.csv` | e383030 (twin c82b120), same path | None of the four. Pair of the row above. | Row `paged,1100,16,32`: `throughput` 495.3, `throughput_cv` 6.63, `tpot_ms` 25.403. **Caveat:** the TPOT fix 0d029d6 landed between 13fbfae and e383030, so the two rows were not produced by the same code. A single-commit check (`results/block_size_sweep/noise_check.json`, 8251fad, two back-to-back runs of this cell) gave 1456.6 and 1232.3 tok/s, a 1.18× spread against the historical 2.19×. |
| `wddm_spill.json` | produced by `python -m experiments.wddm_spill` at 8251fad (dirty only from untracked files), 2026-10-15 | 4. WDDM spill | With no software cap, the FP16 cell `8192×64` finishes with `status` "ok" and `peak_allocated_mb` 14008.6, above `meta.physical_memory_mb` 12226.6 (`peak_reserved_mb` 16270.0, `wall_s` 88.91 against 26.24 for INT8). `16384×32` FP16 gives 14008.7. `meta.driver_model` is "WDDM", `meta.driver_version` 577.13. This reproduces the 14009 MB first-pass figure quoted in `experiments/oom_frontier.py`. |

Notes:

- Pitfall 2 signature: before the fix, per-token latency grows with batch and stays flat across
  seq_len, because the cost scales with the number of Python loop iterations (batch × n_layers
  cache ops per token), not with attention work. In `exp1_layout_comparison_e383030.csv`, paged
  `tpot_ms` at seq 128 is 4.920 (batch 1) → 23.228 (batch 16), ×4.7; at batch 1 it is 4.920
  (seq 128) → 5.093 (seq 2048), ×1.04. Contiguous shows the same direction, weaker: 3.795 → 6.612
  across batch, 3.795 → 3.734 across seq_len. (Batch 16 at seq 2048 is `slow`, not measured.)
  After the fix, both caches are flat in both directions (contiguous 2.92 to 7.07 ms, paged 8.91
  to 11.21 ms), which is genuine per-step overhead at this model size.
- `exp1_layout_comparison1.csv` (2a86eeb) was deleted in the cleanup. It is still in git history.
