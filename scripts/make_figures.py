"""Make the paper's three figures from the committed results. No GPU needed.

  fig1_memory_slope   measured vs predicted per-token memory slope, six architectures,
                      cache term uncorrected and corrected (x1040/1024)
                      <- results/arch_sweep/arch_sweep.json
  fig2_cache_share    KV cache share of peak memory before / after the LM-head fix
                      (FP16, batch 16; logic ported from the old analysis/make_paper_figures.py)
                      <- results/pre_fix/exp3_precision_sweep_017b63a.csv (before)
                         results/precision_sweep/exp3_precision_sweep.csv  (after)
  fig3_oom_frontier   FP16 / INT8 status under enforced memory budgets
                      <- results/oom_frontier/oom_frontier.json

Writes figures/<name>.pdf and .png and nothing else. Numbers behind each figure go to stdout.

Run:  python scripts/make_figures.py
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.memory_model import allocated_slots, arch_sweep_config, predict_slopes  # noqa: E402

# Activation coefficients for figure 1: integer, 3 * d_model + 2 * d_ff (the five tensors live at
# the peak; see scripts/check_memory_model.py and experiments/activation_census.py).
COEFFS = "integer"
HELD_OUT = "12L/768d/3072ff"   # GPT-2 small: the shape not among the five sweep shapes

RESULTS = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "figures")
FIGURES = ("fig1_memory_slope", "fig2_cache_share", "fig3_oom_frontier")

# Validated two-slot categorical palette (CVD dE 24.7, normal-vision dE 33.6 on #fcfcfb).
BLUE, ORANGE = "#2a78d6", "#eb6834"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"          # status: ok / OOM, always with a text label
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "text.color": INK,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    # Fixed metadata so reruns on the same data produce the same files.
    fig.savefig(os.path.join(OUT, stem + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(os.path.join(OUT, stem + ".png"), metadata={"Software": None})
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf, .png")


# ----------------------------------------------------------------------------- figure 1
def figure1():
    print("\n=== fig1_memory_slope <- results/arch_sweep/arch_sweep.json ===")
    arch = json.load(open(os.path.join(RESULTS, "arch_sweep", "arch_sweep.json")))
    L, D, P = arch_sweep_config()
    slots = allocated_slots(L, D, P)
    _, unc = predict_slopes(arch, corrected=False, coeffs=COEFFS)
    _, cor = predict_slopes(arch, corrected=True, coeffs=COEFFS)
    df = pd.DataFrame({
        "shape": [r["shape"] for r in cor], "held_out": [r["shape"] == HELD_OUT for r in cor],
        "measured": [r["measured"] for r in cor],
        "pred_uncorrected": [r["predicted"] for r in unc], "pred_corrected": [r["predicted"] for r in cor],
        "err_uncorrected_pct": [r["error_pct"] for r in unc], "err_corrected_pct": [r["error_pct"] for r in cor],
    }).sort_values("measured").reset_index(drop=True)
    print(f"  cache term over ceil(({L}+{D})/{P})*{P} = {slots} slots per {L} tokens; coefficients {COEFFS}")
    print(df.round(3).to_string(index=False))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})

    lo, hi = 15, 57
    axL.plot([lo, hi], [lo, hi], color=INK_3, lw=1, ls="--", zorder=1, label="predicted = measured")
    fit, held = df[~df.held_out], df[df.held_out]
    axL.scatter(fit.measured, fit.pred_corrected, s=42, color=BLUE, edgecolor=SURFACE, lw=1.5,
                zorder=3, label="sweep shapes")
    axL.scatter(held.measured, held.pred_corrected, s=42, facecolor=SURFACE, edgecolor=BLUE, lw=2,
                zorder=3, label="held out (GPT-2 small)")
    for r in df.itertuples():
        axL.annotate(r.shape, (r.measured, r.pred_corrected), xytext=(5, -9 if r.held_out else 4),
                     textcoords="offset points", fontsize=7, color=INK_2)
    axL.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="measured slope (KB per token)",
            ylabel="predicted slope (KB per token)", title="Model (3·d_model + 2·d_ff) vs measurement")
    axL.set_aspect("equal")
    axL.legend(loc="upper left", fontsize=8)

    y = np.arange(len(df))
    h = 0.36
    axR.barh(y + h / 2, df.err_uncorrected_pct, height=h - 0.04, color=ORANGE,
             label="uncorrected cache term (2·L·d·2 B)")
    axR.barh(y - h / 2, df.err_corrected_pct, height=h - 0.04, color=BLUE,
             label=f"corrected (×{slots}/{L})")
    axR.axvline(0, color=INK_3, lw=0.8)
    for yi, r in zip(y, df.itertuples()):
        for val, off in ((r.err_uncorrected_pct, h / 2), (r.err_corrected_pct, -h / 2)):
            if abs(val) < 0.005:
                axR.text(0.03, yi + off, "0.00%", va="center", ha="left", fontsize=7, color=INK_2)
            else:
                axR.text(val + (0.02 if val >= 0 else -0.02), yi + off, f"{val:+.2f}%",
                         va="center", ha="left" if val >= 0 else "right", fontsize=7, color=INK_2)
    axR.set_yticks(y, [s + ("  (held out)" if ho else "") for s, ho in zip(df["shape"], df.held_out)])
    lim = max(df.err_uncorrected_pct.abs().max(), df.err_corrected_pct.abs().max())
    axR.set_xlim(-lim * 1.35 if df.err_uncorrected_pct.min() < 0 else -0.3, max(0.3, lim * 0.6))
    axR.grid(axis="y", visible=False)
    axR.set(xlabel="prediction error (% of measured slope)", title="Error by architecture")
    axR.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, "fig1_memory_slope")


# ----------------------------------------------------------------------------- figure 2
def cache_share(path, batch=16):
    """FP16 cache_memory_mb / peak_memory_mb at one batch size, by seq_len (ok cells only).
    Same selection as make_paper_figures.py figure3 panel (b)."""
    d = pd.read_csv(path)
    f = d[(d.cache_type == "paged_fp16") & (d.batch == batch)].set_index("seq_len").sort_index()
    share = (100 * f.cache_memory_mb / f.peak_memory_mb).where(f.status == "ok")
    return f.assign(share_pct=share)


def figure2():
    before_p = os.path.join(RESULTS, "pre_fix", "exp3_precision_sweep_017b63a.csv")
    after_p = os.path.join(RESULTS, "precision_sweep", "exp3_precision_sweep.csv")
    print("\n=== fig2_cache_share <- pre_fix/exp3_precision_sweep_017b63a.csv, "
          "precision_sweep/exp3_precision_sweep.csv (paged_fp16, batch 16) ===")
    b, a = cache_share(before_p), cache_share(after_p)
    tbl = pd.DataFrame({"before_status": b.status, "before_cache_mb": b.cache_memory_mb,
                        "before_peak_mb": b.peak_memory_mb, "before_share_pct": b.share_pct,
                        "after_cache_mb": a.cache_memory_mb, "after_peak_mb": a.peak_memory_mb,
                        "after_share_pct": a.share_pct})
    print(tbl.round(1).to_string())

    seqs = list(a.index)
    x = np.arange(len(seqs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.bar(x - w / 2, b.share_pct.reindex(seqs).fillna(0), w - 0.03, color=ORANGE,
           label="before fix: LM head over all prefill positions (017b63a)")
    ax.bar(x + w / 2, a.share_pct, w - 0.03, color=BLUE, label="after fix: LM head on last position only")
    for xi, s in zip(x, seqs):
        bv, st = b.share_pct.get(s), b.status.get(s)
        if pd.notna(bv):
            ax.text(xi - w / 2, bv + 1.2, f"{bv:.0f}%", ha="center", fontsize=8, color=INK_2)
        else:
            ax.text(xi - w / 2, 1.2, str(st).upper(), ha="center", fontsize=7, color=INK_2, rotation=90)
        ax.text(xi + w / 2, a.share_pct[s] + 1.2, f"{a.share_pct[s]:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [str(s) for s in seqs])
    ax.set(xlabel="sequence length (batch 16, FP16 paged cache)", ylabel="KV cache share of peak memory (%)",
           ylim=(0, 70))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=8)
    fig.tight_layout()
    save(fig, "fig2_cache_share")


# ----------------------------------------------------------------------------- figure 3
def figure3():
    print("\n=== fig3_oom_frontier <- results/oom_frontier/oom_frontier.json ===")
    rows = json.load(open(os.path.join(RESULTS, "oom_frontier", "oom_frontier.json")))
    fr = pd.json_normalize(rows)
    fr["cell"] = fr.seq_len.astype(str) + "×" + fr.batch.astype(str)
    cells = list(dict.fromkeys(fr.cell))
    budgets = sorted(fr.budget_gb.unique())
    print(fr.pivot(index="budget_gb", columns="cell", values="verdict")[cells].to_string())
    print(f"  GPU: {', '.join(sorted(fr['gpu'].dropna().unique())) if 'gpu' in fr else 'not recorded'}")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
    for ax, dt in zip(axes, ("fp16", "int8")):
        ax.grid(False)
        for i, bgt in enumerate(budgets):
            for j, c in enumerate(cells):
                r = fr[(fr.budget_gb == bgt) & (fr.cell == c)].iloc[0]
                ok = r[f"{dt}.status"] == "ok"
                frontier = r.verdict == "FRONTIER"
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, lw=0,
                                           facecolor=GOOD if ok else CRITICAL, alpha=0.18 if ok else 0.85))
                if frontier and dt == "int8":
                    ax.add_patch(plt.Rectangle((j - 0.44, i - 0.42), 0.88, 0.84, fill=False,
                                               edgecolor=INK, lw=1.8))
                label = f"ok\n{r[f'{dt}.peak_mb'] / 1024:.1f} GB" if ok else "OOM"
                ax.text(j, i, label, ha="center", va="center", fontsize=7.5,
                        color=INK if ok else "white", fontweight="normal" if ok else "bold")
        ax.set_xlim(-0.5, len(cells) - 0.5)
        ax.set_ylim(len(budgets) - 0.5, -0.5)
        ax.set_xticks(range(len(cells)), [c.replace("×", "\n× ") for c in cells])
        ax.set_yticks(range(len(budgets)), [f"{b:g} GB" for b in budgets])
        ax.set(title=f"{dt.upper()} KV cache", xlabel="seq_len × batch")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("enforced memory budget")
    n_fr = int((fr.verdict == "FRONTIER").sum())
    fig.tight_layout()
    fig.text(0.5, -0.03, f"Cell text: status and peak allocated memory. Outlined: FP16 runs out of "
             f"memory and INT8 completes ({n_fr} budget-cell pairs).", ha="center", fontsize=8, color=INK_2)
    save(fig, "fig3_oom_frontier")


def main():
    figure1()
    figure2()
    figure3()
    stray = sorted(set(os.listdir(OUT)) - {f"{s}.{e}" for s in FIGURES for e in ("pdf", "png")})
    if stray:
        sys.exit(f"figures/ contains files this script does not make: {stray}")
    print("\nDone: figures/ holds exactly the three figures.")


if __name__ == "__main__":
    main()
