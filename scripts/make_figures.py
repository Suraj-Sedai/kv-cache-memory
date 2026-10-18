"""Make the paper's three figures from the committed results. No GPU needed.

  fig1_memory_slope   measured vs predicted per-token memory slope, six architectures
                      (GPT-2 small held out, integer coefficients), and the prediction error
                      with the cache term uncorrected and corrected (x1040/1024)
                      <- results/arch_sweep/arch_sweep.json
  fig2_cache_share    KV cache share of peak memory before / after the LM-head fix
                      (FP16 paged, batch 16)
                      <- results/pre_fix/exp3_precision_sweep_017b63a.csv (before)
                         results/precision_sweep/exp3_precision_sweep.csv  (after)
  fig3_oom_frontier   FP16 / INT8 status under enforced memory budgets
                      <- results/oom_frontier/oom_frontier.json

Print format for arXiv: fixed page size (3.3 in single column, 6.9 in double column; no
bbox_inches="tight", so the PDF page is exactly the figure size), text 7-8 pt at that size, no
titles or caption text inside the plot (captions are in the paper), TrueType (Type 42) fonts
embedded, no PDF date metadata. Matplotlib is pinned in requirements.txt so reruns are
byte-identical.

Writes figures/<name>.pdf (for the paper) and .png (for the README) and nothing else. Numbers
behind each figure go to stdout.

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

SINGLE, DOUBLE = 3.3, 6.9      # column widths, inches
SMALL = 7                      # pt: tick labels, annotations, legends (the minimum used)

# Validated two-slot categorical palette (CVD dE 24.7, normal-vision dE 33.6 on white).
BLUE, ORANGE = "#2a78d6", "#eb6834"
# Figure 3 status: completed cells are a pale tint, OOM cells near-black with white text, so the
# two states differ in lightness (safe in grayscale and for every CVD type) and carry a text label.
OK_FILL, OOM_FILL = "#dce9f8", "#3a3936"
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "legend.fontsize": SMALL,
    "axes.edgecolor": INK_3, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False, "text.color": INK,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
})


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    # No bbox_inches="tight": the page is exactly figsize. Fixed metadata so reruns on the same
    # data produce the same bytes.
    fig.savefig(os.path.join(OUT, stem + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(os.path.join(OUT, stem + ".png"), metadata={"Software": None})
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf, .png  ({w:g} x {h:g} in)")


def panel_tag(ax, tag):
    ax.text(-0.02, 1.04, tag, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, fontweight="bold", color=INK)


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

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), gridspec_kw={"width_ratios": [1, 1.35]})

    lo, hi = 15, 60
    axL.plot([lo, hi], [lo, hi], color=INK_3, lw=0.8, ls="--", zorder=1, label="y = x")
    fit, held = df[~df.held_out], df[df.held_out]
    axL.scatter(fit.measured, fit.pred_corrected, s=22, color=BLUE, edgecolor="white", lw=1,
                zorder=3, label="sweep shapes")
    axL.scatter(held.measured, held.pred_corrected, s=22, facecolor="white", edgecolor=BLUE, lw=1.4,
                zorder=3, label="held out (GPT-2)")
    # Labels below-right of the diagonal (the legend holds the top-left), except: 16L sits 3 KB
    # from 8L/1024d, so it goes above-left; GPT-2 small goes directly below its point.
    for r in df.itertuples():
        if r.held_out:
            off, ha, va = (0, -6), "center", "top"
        elif r.shape == "16L/512d/2048ff":
            off, ha, va = (-5, 2), "right", "bottom"
        else:
            off, ha, va = (5, -2), "left", "top"
        axL.annotate(r.shape, (r.measured, r.pred_corrected), xytext=off, textcoords="offset points",
                     fontsize=SMALL, color=INK_2, ha=ha, va=va)
    axL.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="measured slope (KB per token)",
            ylabel="predicted slope (KB per token)")
    axL.set_aspect("equal")
    axL.legend(loc="upper left", handlelength=1.5, borderaxespad=0.2)
    panel_tag(axL, "a")

    y = np.arange(len(df))
    h = 0.38
    axR.barh(y + h / 2, df.err_uncorrected_pct, height=h - 0.06, color=ORANGE,
             label="uncorrected cache term (2·L·d·2 B)")
    axR.barh(y - h / 2, df.err_corrected_pct, height=h - 0.06, color=BLUE,
             label=f"corrected (×{slots}/{L})")
    axR.axvline(0, color=INK_3, lw=0.6)
    for yi, r in zip(y, df.itertuples()):
        for val, off in ((r.err_uncorrected_pct, h / 2), (r.err_corrected_pct, -h / 2)):
            txt = "0.00%" if abs(val) < 0.005 else f"{val:+.2f}%".replace("-", "−")
            ax_x = 0.03 if abs(val) < 0.005 else val + (0.03 if val >= 0 else -0.03)
            axR.text(ax_x, yi + off, txt, va="center", ha="left" if val >= 0 or abs(val) < 0.005 else "right",
                     fontsize=SMALL, color=INK_2)
    axR.set_yticks(y, [s + (" (held out)" if ho else "") for s, ho in zip(df["shape"], df.held_out)])
    lim = max(df.err_uncorrected_pct.abs().max(), df.err_corrected_pct.abs().max())
    axR.set_xlim(-lim * 1.4 if df.err_uncorrected_pct.min() < 0 else -0.3, max(0.35, lim * 0.35))
    axR.set_ylim(-0.6, len(df) - 0.4 + 1.1)        # headroom for the legend above the bars
    axR.grid(axis="y", visible=False)
    axR.tick_params(axis="y", length=0)
    axR.set(xlabel="prediction error (% of measured slope)")
    axR.legend(loc="upper left", ncol=1, handlelength=1.2, borderaxespad=0.2)
    panel_tag(axR, "b")
    save(fig, "fig1_memory_slope")


# ----------------------------------------------------------------------------- figure 2
def cache_share(path, batch=16):
    """FP16 cache_memory_mb / peak_memory_mb at one batch size, by seq_len (ok cells only)."""
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
    w = 0.4
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    ax.bar(x - w / 2, b.share_pct.reindex(seqs).fillna(0), w - 0.04, color=ORANGE,
           label="before: LM head on all prefill positions")
    ax.bar(x + w / 2, a.share_pct, w - 0.04, color=BLUE, label="after: LM head on last position")
    for xi, s in zip(x, seqs):
        bv, st = b.share_pct.get(s), b.status.get(s)
        if pd.notna(bv):
            ax.text(xi - w / 2, bv + 1.0, f"{bv:.0f}", ha="center", fontsize=SMALL, color=INK_2)
        else:
            ax.text(xi - w / 2, 1.0, str(st).upper(), ha="center", va="bottom", fontsize=SMALL,
                    color=INK_2, rotation=90)
        ax.text(xi + w / 2, a.share_pct[s] + 1.0, f"{a.share_pct[s]:.0f}", ha="center", fontsize=SMALL, color=INK)
    ax.set_xticks(x, [str(s) for s in seqs])
    ax.tick_params(axis="x", length=0)
    ax.set(xlabel="sequence length (batch 16)", ylabel="KV cache share of peak (%)", ylim=(0, 80))
    ax.set_yticks(range(0, 61, 20))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", handlelength=1.2, borderaxespad=0.1)
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
    print(f"  FRONTIER cells (FP16 OOM, INT8 ok): {int((fr.verdict == 'FRONTIER').sum())}")

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), sharey=True)
    for ax, dt, tag in zip(axes, ("fp16", "int8"), ("a", "b")):
        ax.grid(False)
        for i, bgt in enumerate(budgets):
            for j, c in enumerate(cells):
                r = fr[(fr.budget_gb == bgt) & (fr.cell == c)].iloc[0]
                ok = r[f"{dt}.status"] == "ok"
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.44), 0.94, 0.88, lw=0,
                                           facecolor=OK_FILL if ok else OOM_FILL))
                if r.verdict == "FRONTIER" and dt == "int8":
                    ax.add_patch(plt.Rectangle((j - 0.44, i - 0.41), 0.88, 0.82, fill=False,
                                               edgecolor=INK, lw=1.4))
                label = f"ok\n{r[f'{dt}.peak_mb'] / 1024:.1f} GB" if ok else "OOM"
                ax.text(j, i, label, ha="center", va="center", fontsize=SMALL, linespacing=1.15,
                        color=INK if ok else "white", fontweight="normal" if ok else "bold")
        ax.set_xlim(-0.5, len(cells) - 0.5)
        ax.set_ylim(len(budgets) - 0.5, -0.5)
        ax.set_xticks(range(len(cells)), cells)
        ax.set_yticks(range(len(budgets)), [f"{b:g} GB" for b in budgets])
        ax.tick_params(length=0)
        ax.set_xlabel(f"seq_len × batch, {dt.upper()} KV cache")
        for s in ax.spines.values():
            s.set_visible(False)
        panel_tag(ax, tag)
    axes[0].set_ylabel("enforced memory budget")
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
