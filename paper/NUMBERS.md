# Paper numbers → source

Every number in the draft (`paper.md`, kept local and gitignored; line numbers refer to the
version saved 2026-10-15), traced to the committed file and the key, row or column it comes
from. The scripts named here recompute each value from that file without a GPU.

Status:
- **ok**: reproduces from the named source
- **derived**: arithmetic on configuration constants or committed values, shown
- **external**: a citation or a published model fact, not a measurement here

Abbreviations:
- `arch` = `results/arch_sweep/arch_sweep.json`
- `exp3` = `results/precision_sweep/exp3_precision_sweep.csv` (decode length 60)
- `exp3-pre` = `results/pre_fix/exp3_precision_sweep_017b63a.csv`
- `exp1` = `results/cache_comparison/exp1_layout_comparison.csv`
- `exp1-pre` = `results/pre_fix/exp1_layout_comparison_017b63a.csv`
- `oom` = `results/oom_frontier/oom_frontier.json` (decode length 8)
- `ppl` = `results/perplexity_eval/exp3_perplexity.csv`
- `wddm` = `results/pre_fix/wddm_spill.json`
- `census` = `results/activation_census/activation_census.json`
- `T<n>` = `scripts/make_tables.py` Table n
- `CMM` = `scripts/check_memory_model.py`

The memory model uses the integer activation coefficients, 3·d_model + 2·d_ff: the five tensors
live at the peak, counted in `census`. The general form is max(7·d_model, 3·d_model + 2·d_ff);
every tested shape has d_ff ≥ 2·d_model.

## Untraced: 0

Every printed number traces to a committed file. The last two were closed by the GPU runs at
feb71b7: `results/matmul_peak/matmul_peak.json` (decode arithmetic 0.03–0.65%) and
`results/sdpa_backend_check/sdpa_backend_check.json` (32 MB vs 4,888 MB).

## Abstract and Introduction (lines 11–29)

| Line | Number | Source | Key / row / column | Status |
|---|---|---|---|---|
| 11, 19 | within 0.03% (GPT-2 small) | `arch` | gpt2-small `slope_kb_per_token` 53.078 vs predicted 53.0625 → −0.029% | ok; CMM |
| 11, 21 | 5% to 58% | `exp3` | `paged_fp16` `cache_memory_mb / peak_memory_mb`: 512×1 5.24%, 8192×32 58.04% | ok; T1 |
| 11, 21 | 2.7% / 26% | `exp3` | (fp16 − int8 `peak_memory_mb`) / fp16: 2.66%, 26.03% | ok; T1 |
| 11, 29 | +36% capacity | `exp3` | polyfit slopes 27.1526 / 19.9253 KB/token → +36.3% | ok; T2 |
| 21 | INT8 shrinks the cache 48% | `exp3` | 1 − int8/fp16 `cache_memory_mb` = 48.4% in all 20 cells | ok; T1 |
| 25 | 0.08% across six architectures | `arch` | max \|error\| 0.083% (16L/512d/2048ff) | ok; CMM |

## 2. Method (lines 35–51)

| Line | Number | Source | Key / row / column | Status |
|---|---|---|---|---|
| 35 | 8 layers, 8 heads, d_model 512, d_ff 2048, vocab 50,257 | runners | `N_LAYERS, N_HEADS, D_MODEL, D_FF`; `vocab_size=50257` | ok |
| 35 | 11.9 GB | `wddm` | `meta.physical_memory_mb` 12226.6 = 11.94 GiB | ok |
| 35 | PyTorch 2.11 | `wddm`, `*.env.json` | `torch` 2.11.0+cu128 | ok |
| 35 | sequence 1024 (arch sweep) | `experiments/arch_sweep.py` | `SEQ_LEN` | ok |
| 39 | block sizes 1 through 16; 56 tests | `tests/` | `pytest -q` → 56 passed | ok |
| 41 | MB = 2^20 bytes, GB = 2^30 bytes | definition | | — |
| 43, 123 | at 8 GB, 4.9% to 13.4% below the budget | `oom` + `wddm` | ceiling ≥ 7090.6 (16384×16 FP16 ok) and < 7790.6 (8192×48 INT8 OOM); 8 GB = 8192 MB | ok; T7 |
| 45 | decode 60 / 8 | runners | `DECODE_LEN` in `run_precision_sweep.py` / `arch_sweep.py`, `oom_frontier.py` | ok |
| 45 | 8256 vs 8208 slots | derived | ceil((8192+60)/16)·16; ceil((8192+8)/16)·16 | derived |
| 45 | 24 MB | `exp3` / `oom` | 4128.0 − 4104.0 `cache_memory_mb` / `fp16.cache_mb`, 8192×32 | ok |
| 45 | 7113 vs 7086 MB | `exp3` / `oom` | 7112.57 / 7086.5 | ok |
| 47 | 1 timed + 2 warmups, median of 20 | runners | `N_WARMUP = 3`, `N_MEASURE = 20` | ok |
| 49 | amax/127; 512-token windows; 284,614 tokens | `run_perplexity_eval.py`, `ppl` | `fake_quant_per_token`; `ctx` 512; `n_tokens` | ok |
| 51, 169 | 50.8 TFLOP/s; arithmetic 0.03–0.65% of measured decode latency | `results/matmul_peak/matmul_peak.json`; `exp3` | `achievable_tflops` 50.82; decode-step FLOPs / 50.82 TFLOP/s / `tpot_ms` over 20 FP16 cells: 0.03% (1024×1) to 0.65% (1024×32) | ok; T5 |

## 3. A memory model (lines 55–80)

| Line | Number | Source | Key / row / column | Status |
|---|---|---|---|---|
| 55 | 7086 / 7091 MB, 262,144 tokens, 0.058% | `oom` | `fp16.peak_mb` 8192×32 7086.5, 16384×16 7090.6 | ok; notebook 04 |
| 55 | 524,288 tokens within 0.01% | `wddm` | FP16 8192×64 14008.6 vs 16384×32 14008.7 → 0.0007% (`oom` has only INT8 here: 0.038%) | ok |
| 55 | 32 MB (EFFICIENT, CUDNN) vs 4,888 MB (MATH), 2048×16 | `results/sdpa_backend_check/sdpa_backend_check.json` | `backends[].peak_workspace_mb`: 32.0, 32.0 vs 4888.4 | ok |
| 57 | 27.25 KB, 161 MB, 2.42 MB, 1.5–4.5 GB | `arch` | baseline `slope_kb_per_token`, `intercept_mb`, `max_resid_mb`, `peaks` | ok |
| 57 | 146 MB weights | derived | parameters × 2 B without the position table = 146.2 MiB (147.2 with it) | derived |
| 57 | ~15 MB overhead | derived | 161 − 146 | derived |
| 59 | 16.00 KB; 1032 → 1040; 1.015625; 16.25 KB | derived | 2·8·512·2 B; ceil((1024+8)/16)·16 (`src/memory_model.py`) | derived; CMM |
| 61 | 11.01 / 11.00 / 11.04 KB | `arch` | slope − corrected cache: 11.009, 11.000, 11.036 | ok; CMM |
| 61 | 14.03, 18.99 | `arch` | 14.030, 18.988 | ok; CMM |
| 63 | 3.03 and 2.00 tensors | `arch` | free fit a = 3.030, b = 1.997 | ok; CMM (validation) |
| 65 | 3·d_model + 2·d_ff | `census` | `live_d_model_tensors` 3, `live_d_ff_tensors` 2 in all six shapes | ok |
| 69 | 53.062, 53.078, 0.03%, 16 bytes, 0.08% | `arch` | integer prediction 53.0625; 0.0155 KB = 15.9 B | ok; CMM |
| 75–80 | Table 1, every cell | `arch` | CMM "integer coefficients, corrected" block | ok |

## 4. Measurement pitfalls (lines 90–103)

| Line | Number | Source | Key / row / column | Status |
|---|---|---|---|---|
| 90, 95, 103 | 13.3% (13%) of peak before the fix, 2048×16 | `exp3-pre` | `paged_fp16` 528.0 / 3958.11 | ok; T6 |
| 90 | 50.8–58.0%; INT8 saves 22.8–26.0% | `exp3` | 2048×16, 8192×32 | ok; T1 |
| 91, 97 | 4.920 → 23.228 ms (batch 1 → 16, seq 128); 4.920 → 5.093 ms (seq 128 → 2048, batch 1) | `exp1-pre` | paged `tpot_ms` | ok; T9 |
| 91, 97 | batch effect 4.7×; largest unpaired drift 1.76× | `exp1-pre`; `exp1` + `results/paired_layout_check/paired_layout_check.json` | 23.228 / 4.920 = 4.72; 2048×1: interleaved median 0.6524 / unpaired 0.3715 = 1.76 | ok; T9 |
| 97 | 32× cache operations at batch 32 | derived | batch × n_layers vs n_layers | derived |
| 92, 99, 153 | reservation waste 0.0 | `exp1` | contiguous `frag_ratio` 0.0 in all 20 rows | ok |
| 93, 101 | 14,008.6 MB on a 12,226.6 MB card | `wddm` | FP16 8192×64 `peak_allocated_mb`; `meta.physical_memory_mb` | ok |
| 95 | logits 3141 MB | derived | 16 × 2048 × 50257 × 2 B = 3141.1 MiB | derived; T6 |
| 95 | 79% of a 3958 MB peak | `exp3-pre` | 3141.1 / 3958.11 = 79.4% | ok; T6 |
| 95 | 1039 MB; 13.3% → 50.8% | `exp3-pre` → `exp3` | 2048×16 | ok; T6 |

## 5. Results (lines 107–145)

| Line | Number | Source | Key / row / column | Status |
|---|---|---|---|---|
| 111 | 9 MB of 172 MB, 5.2%; 35.5%; 50.8%; 4128 of 7113 MB, 58.0% | `exp3` | `paged_fp16` 512×1, 1024×8, 2048×16, 8192×32 | ok; T1 + reported checks |
| 113 | 48.4%; 2.7 / 15.9 / 22.8 / 26.0% | `exp3` | | ok; T1 |
| 115, 157 | 13.3% at 2048×16; 4.1% (512×1) to 13.8% (512×32) | `exp3-pre` | `paged_fp16` share, 16 ok cells | ok; T6 |
| 115 | INT8 reduced peak "a few percent at best" (before the fix) | `exp3-pre` | 2.4–6.7% over 16 cells | ok |
| 119 | 9 / 10 GB: 8192×48 FP16 OOM, INT8 ok at 7791 MB | `oom` | `int8.peak_mb` 7790.6 | ok |
| 119 | 11 / 11.5 GB: 8192×64, 16384×32 | `oom` | `verdict` FRONTIER | ok; fig 3 |
| 119 | 14009 MB FP16 | `wddm` | 14008.6 / 14008.7 | ok |
| 119 | ~10,330 MB INT8 | `oom` | 10328.9 / 10332.8 | ok |
| 119 | six cells across four budgets | `oom` | 6 FRONTIER rows; budgets 9, 10, 11, 11.5 | ok; fig 3 |
| 121 | 33/64 = 0.515625 at head_dim 64 | derived | int8 codes + FP16 scale per token per head per layer, `int8_paged.py:23–27`; head_dim = 512/8 | derived; CMM |
| 121 | 0.516 in every cell | `exp3` | int8/fp16 `cache_memory_mb` 0.5156–0.5159 | ok; CMM |
| 123 | 7791 MB fails at 8 GB; 16384×16 FP16 completes at 7091 MB | `oom` | budget 8.0 rows | ok |
| 127 | 27.1526 KB + 168.0 MB; 19.9253 KB + 164.0 MB; 27.25 / 161 | `exp3`; `arch` | polyfit over 20 cells per dtype | ok; T2 |
| 129 | 303k / 413k (8 GB); 457k / 623k (12 GB); 36.3%; 4 MB; 1.3627 | derived from the fits | 302,606 / 412,574; 457,078 / 623,076 | ok |
| 131 | 1538k / 2097k (40 GB); 3083k / 4202k (80 GB) | derived from the fits | 1,538,378 / 2,096,586; 3,083,091 / 4,201,602 | ok (extrapolation) |
| 135 | 89.6–93.4%, median 92.5% (19 cells); batch 1 89.6–91.5%; batch ≥ 8 90.9–93.4% | `exp3` | Δ`peak_memory_mb` / Δ`cache_memory_mb`, excluding 512×1 | ok |
| 135 | 512×1: 4.57 MB vs 4.36 MB, above 100% | `exp3` | 171.88 − 167.31; 9.0 − 4.64 (the INT8 peak reruns at 168.31, within the 1 MB tolerance) | ok |
| 139 | paged/contiguous throughput median 0.58–0.65 (512×1, 2048×1, 2048×16; 5 interleaved pairs each; paged slower in 15/15) | `results/paired_layout_check/paired_layout_check.json` | `cells[].median_ratio` 0.5894, 0.6524, 0.5838; `pairs[].ratio` all < 1 | ok; T9 |
| 139 | paged/contiguous peak within 1% | `exp1` | max 0.9% | ok; T4 |
| 143 | 98.54% / 98.83%; +0.002% / +0.006%; ~1.5% flips; 284,614 | `ppl` | rows 1–2 `top1_agreement`, `ppl_pct_increase`, `n_tokens` | ok; T3 |
| 145, 167 | 355M parameters | GPT-2-medium | | external |

## 6. Withdrawn claims (lines 151–157)

| Line | Number | Source | Key / row / column | Status |
|---|---|---|---|---|
| 151 | blocks of 8–16 favored | `results/pre_fix/exp2_block_size_sweep_13fbfae.csv` | blocks 8 and 16 win all 15 (seq_len, batch) cells; mean 314.4 / 326.6 vs 258.3 / 181.1 tok/s | ok; T8 |
| 155 | 20 cells, seq 128–2048, batch 1–16; paged slower in all 20 unpaired cells and all 15 interleaved pairs; 0.9% | `exp1`; `results/paired_layout_check/paired_layout_check.json` | | ok; T4, T9 |
| 157 | 58% | `exp3` | 8192×32 | ok |

## 7. Limitations and 8. Related work (lines 161–185)

| Line | Number | Source | Key / row / column | Status |
|---|---|---|---|---|
| 165 | 11.9 GB | `wddm` | | ok |
| 169 | same commit, 1.18× throughput | `results/block_size_sweep/noise_check.json` | `max_over_min` 1.182 (1456.6 vs 1232.3) | ok |
| 175 | 133 systems papers | [13] | | external |
| 177 | vLLM default block size 16 | [10] | | external |
| 179 | FlashAttention unavailable in this PyTorch build | `results/sdpa_backend_check/sdpa_backend_check.json` | FLASH_ATTENTION `available` false, "No available kernel"; run warning "Torch was not compiled with flash attention" | ok |
| 183 | 2.7% or 26% | `exp3` | | ok |
| 185 | "roughly 1.2 times model memory" | [2] | | external |
| 185 | 11.01, 11.00, 11.04 KB at 4/8/16 layers; 0.08% | `arch` | | ok; CMM |

## Not printed, still open

| Item | Status |
|---|---|
| exp1's unpaired paged/contiguous range 0.32–0.77× | Not printed. The paper uses the interleaved check instead (line 139); exp1's per-cell ratios differ from it by up to 1.76× (T9) |
| Editor's note "exp1 reports 1.35–2.3×" | no committed file gives it |
