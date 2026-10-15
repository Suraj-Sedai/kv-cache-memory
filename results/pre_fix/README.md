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
| `exp1_layout_comparison_e383030.csv` | e383030 (twin c82b120), `experiments/results/exp1_layout_comparison.csv` | 2. Per-sequence loop (also 1 and 3) | **Pitfall 2:** paged `tpot_ms` scales with batch and not with seq_len: 4.92 / 8.04 / 13.18 / 23.23 ms at seq 128, batch 1 / 4 / 8 / 16, and 23.23 / 22.82 / 23.68 / 26.45 at batch 16, seq 128 to 1024. Post-fix paged is 10.62 to 11.21 at seq 128 for every batch. **Pitfall 1:** `peak_memory_mb` is 332.95 (contiguous, 128×1) against 161.27 post-fix, with `cache_memory_mb` 5.88. **Pitfall 3:** contiguous `frag_ratio` is 0.0 in all 19 `ok` rows. |
| `exp1_layout_comparison_0584aa1.csv` | 0584aa1, `experiments/results/exp1_layout_comparison2.csv` (old 9-column schema) | 3. Oracle reservation | Contiguous `frag_ratio` is exactly 0.0 in all 20 contiguous rows; paged is 0.25 in every row. This is the old per-last-block definition of `frag_ratio`, not comparable to current values. |
| `exp2_block_size_sweep_13fbfae.csv` | 13fbfae, `experiments/results/exp2_block_size_sweep.csv` | None of the four. Evidence for the withdrawn block-size claim (noise). | Row `paged,1100,16,32`: `throughput` 226.6, `throughput_cv` 5.52, `tpot_ms` 62.747. |
| `exp2_block_size_sweep_e383030.csv` | e383030 (twin c82b120), same path | None of the four. Pair of the row above. | Row `paged,1100,16,32`: `throughput` 495.3, `throughput_cv` 6.63, `tpot_ms` 25.403. **Caveat:** the TPOT fix 0d029d6 landed between 13fbfae and e383030, so the two rows were not produced by the same code. Use `experiments/block_size_noise.py` to test the claim on a single commit. |
| `wddm_spill.json` | *not yet produced* | 4. WDDM spill | Written by `python -m experiments.wddm_spill`. Before that run, the only record is the comment in `experiments/oom_frontier.py` ("14009 MB allocated as 'ok' on an 11.9 GB device"). |

Notes:

- On pitfall 2, the README describes the symptom as "latency barely moves across 16× changes in batch and seq_len". In
  these files, the **pre-fix** paged TPOT is flat in seq_len but grows about 4.7× with batch. The
  **post-fix** file is the one that is flat in both (contiguous 2.92 to 7.07 ms, paged 8.91 to 11.21 ms).
- `exp1_layout_comparison1.csv` (2a86eeb) was deleted in the cleanup. It is still in git history.
