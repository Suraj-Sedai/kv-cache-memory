# Paper numbers → source

Every number printed in the draft (`paper.md`, kept local), traced to the committed file and the
key, row, or column it comes from. Scripts named here recompute the value from that file without
a GPU.

Status:
- **ok**: reproduces from the named source
- **derived**: arithmetic on configuration constants or committed values, shown
- **FLAG**: does not reproduce, or no committed source exists

Abbreviations:
- `arch` = `results/arch_sweep/arch_sweep.json`
- `exp3` = `results/precision_sweep/exp3_precision_sweep.csv` (decode length 60)
- `exp3-pre` = `results/pre_fix/exp3_precision_sweep_e383030.csv`
- `exp1` = `results/cache_comparison/exp1_layout_comparison.csv`
- `oom` = `results/oom_frontier/oom_frontier.json` (decode length 8)
- `ppl` = `results/perplexity_eval/exp3_perplexity.csv`
- `wddm` = `results/pre_fix/wddm_spill.json`

The memory-model rows use the **integer** activation coefficients (3·d_model + 2·d_ff), which
is the form the paper's Table 1 uses. The fitted coefficients give 53.0715 and 0.012% for GPT-2;
see `scripts/check_memory_model.py`.

## Abstract and Introduction

| Number | Source | Key / row / column | Status |
|---|---|---|---|
| within 0.03% (GPT-2 small) | `arch` | label `gpt2-small`, `slope_kb_per_token` 53.078 vs predicted 53.0625 → −0.029% | ok (integer coefficients); `check_memory_model.py` |
| 5% to 58% cache share | `exp3` | `paged_fp16`: `cache_memory_mb / peak_memory_mb`, 512×1 = 5.24%, 8192×32 = 58.04% | ok; `make_tables.py` Table 1 |
| peak falls 2.7% / 26% | `exp3` | (`peak_memory_mb` fp16 − int8) / fp16 at 512×1 = 2.66%, 8192×32 = 26.03% | ok |
| token capacity +36% | `exp3` | polyfit slopes 27.1526 / 19.9253 KB/token → +36.3% at 8 GB | ok; `make_tables.py` Table 2 |
| INT8 shrinks the cache 48% | `exp3` | 1 − int8/fp16 `cache_memory_mb` = 48.4% in all 20 cells | ok |
| 0.08% across six architectures | `arch` | max \|error\| = 0.083% (16L/512d/2048ff) | ok |

## 2. Method

| Number | Source | Key / row / column | Status |
|---|---|---|---|
| 8 layers, 8 heads, d_model 512, d_ff 2048, vocab 50,257 | `experiments/*.py` | `N_LAYERS, N_HEADS, D_MODEL, D_FF`; `vocab_size=50257` | ok |
| 11.9 GB device memory | `wddm` | `meta.physical_memory_mb` 12226.6 MiB = 11.94 GiB | ok |
| PyTorch 2.11 | `wddm` / any `*.env.json` | `torch` 2.11.0+cu128 | ok |
| sequence length 1024 (arch sweep) | `experiments/arch_sweep.py` | `SEQ_LEN` | ok |
| block sizes 1 through 16; 56 tests | `tests/` | `pytest -q` → 56 passed | ok |
| effective ceiling 5–15% below budget | `oom` | INT8 8192×48 OOM at 8 GB with a 7790.6 MB peak elsewhere = 4.9% below; **the 15% end has no source** | **FLAG** (upper bound) |
| decode 60 / 8 tokens | `run_precision_sweep.py` / `arch_sweep.py`, `oom_frontier.py` | `DECODE_LEN` | ok |
| 8256 vs 8208 slots | derived | ceil((8192+60)/16)·16 = 8256; ceil((8192+8)/16)·16 = 8208 | derived |
| 24 MB cache difference | `exp3` / `oom` | 4128.0 (exp3 8192×32 `cache_memory_mb`) − 4104.0 (`oom` 8 GB 8192×32 `fp16.cache_mb`) | ok |
| peak 7113 vs 7086 MB | `exp3` / `oom` | 7112.57 / 7086.5 | ok |
| 1 timed + 2 warmups, median of 20 | runners | `N_WARMUP = 3`, `N_MEASURE = 20` | ok |
| 284,614 tokens | `ppl` | `n_tokens`, rows 1–2 | ok |
| 512-token windows | `ppl` | `ctx` 512 | ok |
| decode step ~0.7% arithmetic | none | not computed by any committed script | **FLAG** |

## 3. A memory model

| Number | Source | Key / row / column | Status |
|---|---|---|---|
| 8192×32 → 7086 MB, 16384×16 → 7091 MB, 0.058% | `oom` | `fp16.peak_mb` 7086.5, 7090.6 (any budget ≥ 8 GB) | ok; notebook 04 cell 4 |
| 524,288 tokens agree within 0.01% | `wddm` | FP16 8192×64 14008.6 vs 16384×32 14008.7 → 0.0007% (uncapped run). In `oom` only INT8 reaches 524k: 10328.9 vs 10332.8 = 0.038% | ok via `wddm`; **note** the INT8 pair does not meet 0.01% |
| 32 MB (EFFICIENT, CUDNN) vs 4,888 MB (MATH) | none | printed by `benchmarks/sdpa_backend_check.py` (deleted in this cleanup, still at tag `pre-cleanup`); its output was never saved | **FLAG** |
| 27.25 KB/token, 161 MB, 2.42 MB, 1.5–4.5 GB | `arch` | baseline: `slope_kb_per_token`, `intercept_mb`, `max_resid_mb`, `peaks` | ok |
| 146 MB FP16 weights | derived | parameter count ×2 B = 146.2 MiB without the position table, 147.2 with it (notebook 05 uses 147.2) | derived; **note** which count is meant |
| ~15 MB allocator overhead | derived | 161 − 146 | derived |
| 16.00 KB, 1040/1024, 16.25 KB | derived | 2·8·512·2 B; ceil((1024+8)/16)·16 = 1040 (`src/memory_model.py`) | derived |
| 11.01 / 11.00 / 11.04 KB/token | `arch` | `slope_kb_per_token` − corrected cache: 11.009, 11.000, 11.036 | ok |
| 14.03 (2× d_model), 18.99 (2× d_ff) | `arch` | 14.030, 18.988 | ok |
| 3.03 and 2.00 tensors | `arch` | fitted a = 3.030, b = 1.997 (`check_memory_model.py`, fitted) | ok |
| predicts 53.062, measured 53.078, 0.03%, 16 bytes | `arch` | integer: 53.0625; 53.078 − 53.0625 = 0.0155 KB = 15.9 B | ok (integer coefficients) |
| Table 1, all rows | `arch` | `check_memory_model.py`, "integer, corrected" block: errors −0.047, 0.000, −0.083, −0.064, +0.034, −0.029% | ok |

## 4. Measurement pitfalls

| Number | Source | Key / row / column | Status |
|---|---|---|---|
| "KV cache is 13% of peak" (Table 2) | `exp3-pre` | `paged_fp16` 2048×16: 528.0 / 3958.11 = 13.3% | ok |
| 50.8–58.0% at large cells | `exp3` | 2048×16 50.8%, 8192×32 58.0% | ok |
| INT8 saves 22.8–26.0% | `exp3` | peak cut 2048×16 22.8%, 8192×32 26.0% | ok |
| 14009 MB on an 11.9 GB card | `wddm` | FP16 8192×64 `peak_allocated_mb` 14008.6 | ok |
| logits 3141 MB | derived | 16 × 2048 × 50257 × 2 B = 3141.1 MiB | derived |
| 81% of a 3859 MB peak | none | no committed file has 3859 MB at 2048×16. `exp3-pre` gives 3958.11 MB (logits = 79%) | **FLAG** |
| drops peak to 1039 MB | `exp3` / `exp1` | 2048×16 `peak_memory_mb` 1038.9 | ok |
| cache 13.7% → 50.8% | `exp3-pre` → `exp3` | before: 13.7% has no source (528/3859); `exp3-pre` gives 13.3%. After: 50.8% | **FLAG** (before value) |
| prefill falls by 67 ms | none | `ttft_ms` at 2048×16 falls 802.4 → 75.8 ms (726.6 ms) between `exp3-pre` and `exp3` | **FLAG** |
| 1.69 TFLOP removed | derived | 2 × 16 × 2047 × 512 × 50257 = 1.686e12 | derived |
| per-sequence loop: latency flat across 16× batch and sequence | `results/pre_fix/exp1_layout_comparison_e383030.csv` | pre-fix paged `tpot_ms` grows 4.920 → 23.228 ms from batch 1 to 16 (seq 128) and is flat across seq_len (4.920 → 5.093, batch 1). Flat in both only after the fix | **FLAG** (wording; README and `results/pre_fix/README.md` are already corrected) |
| reservation waste 0.0 in every cell | `exp1` | contiguous `frag_ratio` 0.0 in all 20 rows | ok |

## 5. Results

| Number | Source | Key / row / column | Status |
|---|---|---|---|
| 9 MB of 172 MB, 5.2% | `exp3` | `paged_fp16` 512×1: 9.0 / 171.88 | ok |
| 35.5% (1024×8), 50.8% (2048×16) | `exp3` | 35.49%, 50.82% | ok |
| 4128 MB of 7113, 58.0% | `exp3` | 8192×32: 4128.0 / 7112.57 | ok |
| 48.4% in every cell | `exp3` | 48.4% in all 20 | ok |
| 2.7 / 15.9 / 22.8 / 26.0% | `exp3` | 2.66, 15.90, 22.79, 26.03 | ok |
| before the fix "roughly 13% across the grid" (§5.1); "roughly 10–14%" (§6) | `exp3-pre` | `paged_fp16` share is 4.1–13.8% across the 16 `ok` cells (10.5–13.3% at batch 16) | **FLAG** (range) |
| 9 GB: 8192×48 FP16 OOM, INT8 ok at 7791 MB; same at 10 GB | `oom` | budget 9.0 / 10.0, 8192×48: `fp16.status` oom, `int8.peak_mb` 7790.6 | ok |
| 11 and 11.5 GB: 8192×64 and 16384×32 | `oom` | `verdict` FRONTIER | ok |
| 14009 MB FP16 for those cells | `wddm` | 14008.6 / 14008.7 | ok |
| ~10,330 MB INT8 | `oom` | `int8.peak_mb` 10328.9 / 10332.8 | ok |
| six cells across four budgets | `oom` | 6 rows with `verdict` FRONTIER, budgets 9, 10, 11, 11.5 | ok; `make_figures.py` fig 3 |
| (1 + 2/64)/2 = 0.515625 | derived | | derived |
| measured ratio 0.515627–0.515639 | `exp3` | int8/fp16 `cache_memory_mb` gives 0.515556–0.515882; the CSV rounds to 0.01 MB, too coarse for 6 decimals | **FLAG** |
| INT8 8192×48 fails at 8 GB despite 7791 MB | `oom` | budget 8.0, 8192×48: `int8.status` oom | ok |
| 27.1526 KB + 168.0 MB; 19.9253 KB + 164.0 MB | `exp3` | numpy polyfit over 20 `ok` cells per dtype | ok; `make_tables.py` Table 2 |
| 303k vs 413k at 8 GB; 457k vs 623k at 12 GB | derived from fits | 302,606 / 412,574; 457,078 / 623,076 | ok |
| 36.3%; ratio 1.3627 | derived from fits | 36.34% (8 GB), 36.32% (12 GB); 27.1526 / 19.9253 = 1.3627 | ok |
| intercepts differ by 4 MB | derived from fits | 168.02 − 164.02 | ok |
| 1538k vs 2097k at 40 GB; 3083k vs 4202k at 80 GB | derived from fits | 1,538,378 / 2,096,586; 3,083,091 / 4,201,602 | ok (extrapolation) |
| realized saving 89.6–93.4%, median 92.5% (19 cells) | `exp3` | (Δ`peak_memory_mb`) / (Δ`cache_memory_mb`) excluding 512×1 | ok |
| batch 1: 89.6–91.5% | `exp3` | same, batch 1 | ok |
| batch ≥ 8: "consistently 92.5–93.0%" | `exp3` | same, batch ≥ 8: 90.9–93.4% | **FLAG** |
| 512×1: peak −4.57 MB vs cache −4.36 MB | `exp3` | 171.88 − 167.31; 9.0 − 4.64 | ok (the 167.31 reruns at 168.31 on the current environment, so the >100% ratio is within the 1 MB tolerance) |
| paged/contiguous peak within 1% | `exp1` | max 0.9% | ok; `make_tables.py` Table 4 |
| 98.54% / 98.83% same next token | `ppl` | `top1_agreement` rows 1–2 | ok |
| +0.002% / +0.006% perplexity | `ppl` | `ppl_pct_increase` rows 1–2 | ok |
| ~1.5% argmax flips | `ppl` | 1 − 0.9854 = 1.46% (GPT-2) | ok |
| 355M parameters | external | GPT-2-medium | — |

## 6. Withdrawn claims

| Number | Source | Key / row / column | Status |
|---|---|---|---|
| blocks of 8–16 favored | none committed | earlier exp2 ranking; not computed by any current script | **FLAG** (low priority) |
| 226.6 and 495.3 tok/s, factor 2.2, "identical configurations" | `results/pre_fix/exp2_block_size_sweep_13fbfae.csv`, `..._e383030.csv` | row paged,1100,16,32 `throughput` | **FLAG**: the two runs are different commits (TPOT fix 0d029d6 between). On one commit, `results/block_size_sweep/noise_check.json` gives 1456.6 vs 1232.3 (1.18×) |
| 20 cells, seq 128–2048, batch 1–16 | `exp1` | grid | ok |
| peak within 0.9% | `exp1` | | ok |
| cache share reaches 58% | `exp3` | 8192×32 | ok |

## Editor's note (not printed, but the number is open)

| Number | Source | Status |
|---|---|---|
| "exp1 reports 1.35–2.3×" paged throughput | `exp1` gives paged/contiguous 0.32–0.77× (contiguous/paged 1.30–3.13×) over 20 cells; `make_tables.py` Table 4. No committed file gives 1.35–2.3× | **FLAG** |

## Not traced to notebooks 01 or 02

No number in the draft depends on `analysis/notebooks/01_baseline_analysis.ipynb` or
`02_crossover_analysis.ipynb`. The only overlap is the 0.32–0.77 throughput ratio, which
`make_tables.py` computes from `exp1` directly.
