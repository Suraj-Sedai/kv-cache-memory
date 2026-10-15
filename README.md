# Measuring the Wrong Ledger

Where KV cache memory goes during transformer inference, measured on one consumer GPU, and four measurement bugs that each gave me a confident wrong answer first.


**Main point:** a memory optimization only pays off when the thing it shrinks is a large share of memory. INT8 halves the KV cache in every workload here, but that cuts peak memory by anywhere from 2.7% to 26%. The technique doesn't change. The cache's share of memory does.

## INT8 is worth as much as the cache's share of memory

| seq_len | batch | Cache share of peak | INT8 cuts peak by |
|--:|--:|--:|--:|
| 512 | 1 | 5.2% | 2.7% |
| 1024 | 8 | 35.5% | 15.9% |
| 2048 | 16 | 50.8% | 22.8% |
| 8192 | 32 | 58.0% | 26.0% |

The INT8 cache is 48.4% smaller in every row. Under a fixed memory budget, 36% more tokens fit (about 303k → 412k at 8 GB). On real GPT-2 weights, perplexity changes by less than 0.01%, but 1.2–1.5% of next-token predictions flip.

## A per-token memory model

Peak memory depends only on total tokens (batch × seq_len). Each token costs:

```
bytes per token = ( 2 · n_layers · d_model · c   +   3 · d_model + 2 · d_ff ) × bytes per element
                    └──────── KV cache ────────┘     └──── activations ────┘
```

The cache is kept for every layer. Activations are not: each layer's temporaries are freed before the next layer runs, so the activation term stays flat as the model gets deeper (11.01, 11.00, 11.04 KB/token at 4, 8, 16 layers). `c = 1040/1024` accounts for decode tokens and 16-token pages.

Built from five model shapes, the formula predicts GPT-2 small, a shape it never saw, at 53.06 KB/token. Measured: 53.08 (0.03% off). Worst error across all six shapes: 0.08%.

## Four bugs, four wrong conclusions

None raised an error. Each returned a believable number.

| Bug | What I wrongly concluded | How to spot it in your own setup |
|---|---|---|
| LM head ran on every prefill position, not just the last | KV cache is at most 14% of peak, so INT8 can't help | Measured peak is bigger than your hand-computed sum of tensors |
| Attention looped over the batch in Python | Decode is launch-bound on this GPU | Per-token latency grows with batch but stays flat across seq_len: cost scales with Python loop iterations, not attention work. Before the fix, paged TPOT at seq 128 went 4.92 → 23.23 ms from batch 1 to 16, and at batch 1 went 4.92 → 5.09 ms from seq 128 to 2048 |
| Contiguous cache reserved exactly the final length | Contiguous never fragments | Fragmentation is exactly 0 in every cell |
| Windows driver spilled to host RAM instead of failing | This GPU never runs out of memory | 14,009 MB "allocated" on an 11.9 GB card |

## What didn't hold up

- **Paged beats contiguous somewhere.** Not in any of 20 cells: paged throughput is 0.32–0.77× contiguous, and peak memory is within 0.9%. This setup can't show paging's benefit, because all sequences are the same length and the page pool is allocated at full size up front.
- **Contiguous fragments less than paged.** Only because of bug 3.
- **Small block sizes are fastest.** Two runs of the same config gave 227 and 495 tok/s. Too noisy to rank.

## Setup and limits

- Study model: 8 layers, d_model 512, d_ff 2048, FP16, random weights. Random weights are fine for memory, which depends on tensor shapes, not values. Quality is measured on real GPT-2 and GPT-2-medium weights (WikiText-2, 284,614 tokens).
- One GPU: RTX 5070 Ti Laptop, 11.9 GB, Windows. Memory budgets are enforced in software.
- Standard multi-head attention and MLP only. Grouped-query attention changes the cache term and SwiGLU changes the activation term, so Llama-style models need the coefficients re-derived.
- The paged cache copies blocks into one tensor before attention (no fused kernel), so its speed is not the speed of paging in a real server.
- No latency claims. At this model size a decode step is mostly kernel-launch overhead.

## Repository layout

```
src/                 KV caches (contiguous, paged, INT8 paged), model, profiling, memory_model.py
experiments/         one runner per experiment; every runner takes --help and --seed
results/<runner>/    committed outputs of each runner
results/pre_fix/     outputs from before the four fixes, with a README mapping each to its bug
scripts/             make_figures.py, make_tables.py, check_memory_model.py (no GPU needed)
figures/             written only by scripts/make_figures.py
notebooks/           04 memory model and OOM frontier, 05 architecture sweep
data/                WikiText-2 test set, short eval text
tests/               56 tests; paged and contiguous must produce identical tokens
```

## Reproduce

### From the committed results (no GPU, seconds)

```
pip install -r requirements.txt
```

| What | Command | Reads |
|---|---|---|
| Figures 1–3 | `python scripts/make_figures.py` | `results/arch_sweep`, `results/precision_sweep`, `results/pre_fix`, `results/oom_frontier` |
| Cache share / INT8 table, FP16 and INT8 fits, perplexity, paged vs contiguous | `python scripts/make_tables.py` | `results/precision_sweep`, `results/perplexity_eval`, `results/cache_comparison` |
| Per-token memory model, uncorrected and corrected | `python scripts/check_memory_model.py` | `results/arch_sweep/arch_sweep.json` |
| Tests | `python -m pytest -q` | — |

Both scripts exit non-zero if a reported value does not reproduce. To check a rerun, pass its output with `--against`: `python scripts/make_tables.py --against <rerun>.csv` (CSV runners; the file name must match the committed one) or `python scripts/check_memory_model.py --against <rerun>.json` (arch sweep). Memory columns are compared at ±1 MB.

### Rerunning the experiments (GPU)

Install PyTorch with CUDA first ([pytorch.org](https://pytorch.org/get-started/locally/); tested on 2.11 with CUDA 12.8). Each runner overwrites its file in `results/`; pass `--output` to write elsewhere. Every runner also writes `<output>.env.json` (torch, CUDA, driver, GPU, git commit, package versions); the perplexity runner appends one line per row to `<output>.env.jsonl`. Committed results that predate this have no env file.

| Experiment | Command | Writes |
|---|---|---|
| Memory model across architectures | `python -m experiments.arch_sweep` | `results/arch_sweep/arch_sweep.json` |
| FP16 vs INT8 cache | `python -m experiments.run_precision_sweep` | `results/precision_sweep/exp3_precision_sweep.csv` |
| OOM frontier under budgets | `python -m experiments.oom_frontier` | `results/oom_frontier/oom_frontier.json` |
| Paged vs contiguous | `python -m experiments.run_cache_comparison` | `results/cache_comparison/exp1_layout_comparison.csv` |
| Block-size sweep | `python -m experiments.run_block_size_sweep` | `results/block_size_sweep/exp2_block_size_sweep.csv` |
| INT8 perplexity (one row per call; appends) | `python -m experiments.run_perplexity_eval --model gpt2 --text data/wikitext2_test.txt` | `results/perplexity_eval/exp3_perplexity.csv` |
| Driver spill, no memory cap | `python -m experiments.wddm_spill` | `results/pre_fix/wddm_spill.json` |
| Block-size noise, same commit | `python -m experiments.block_size_noise` | `results/block_size_sweep/noise_check.json` |

Use `--help` on any runner to select a subset of the grid (for example `--seq-lens 512 --batches 1`).

`exp3_perplexity.csv` row 3 is a 1,135-token test run on `data/eval_text.txt`. It is kept and marked in the `excluded` column; do not cite it.

## Hardware and OS caveats

- **One machine.** RTX 5070 Ti Laptop GPU (11.9 GB), driver 577.13, Windows 11, PyTorch 2.11 + CUDA 12.8.
- **Windows WDDM sysmem fallback.** On Windows the NVIDIA driver can spill allocations past VRAM into host RAM over PCIe instead of raising out-of-memory. A run that should fail then "succeeds", only slower. `experiments/wddm_spill.py` records this.
- **The OOM limit is enforced in software.** `oom_frontier.py` and `arch_sweep.py` cap PyTorch's allocator with `torch.cuda.set_per_process_memory_fraction`, so OOM is raised in-process before the driver fallback engages. Results are stated per budget (8–11.5 GB), not per device. The cap counts allocated bytes, not what the driver reserves.
- **Peak memory is `torch.cuda.max_memory_allocated`.** It does not include allocator fragmentation or reserved-but-unused memory.
- **Rerun tolerance.** memory values reproduce to within 1 MB across environments; observed differences are allocator granularity. Rerunning committed cells on this machine (October 2026) reproduced `cache_memory_mb` exactly and `peak_memory_mb` exactly on most cells; the largest difference was INT8 512×1, 168.31 vs 167.31 MB committed. The unmodified original code gives the same new values. Perplexity rows reproduce exactly.
- **Timing columns are not comparable across runs.** Clocks are not locked and cells run in a fixed order. No latency claims are made.
