#!/usr/bin/env python
"""Figure 5 — AUC vs degradation severity (clean + 3 levels) with 95 % CI bands for TAMES AI, TAMES AI with Q_m = 1,
the image-only CNN and the late fusion; mean q̄ on a right axis in panel (a). Input: runs/tn3k_fold0/robustness/robustness.csv.
    python scripts/plots/fig5_robustness.py"""
import sys; sys.path.insert(0, "scripts/plots")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from _style import setup, save, line, band, panel, W2, MM, INK2

setup()
d = pd.read_csv("runs/tn3k_fold0/robustness/robustness.csv")
clean = d[d.degradation == "clean"]
names = {"gaussian_noise": "Gaussian noise ($\\sigma$)", "speckle_noise": "speckle noise ($\\sigma$)", "blur": "blur ($\\sigma$, px)", "contrast": "contrast (factor)"}
fig, axes = plt.subplots(1, 4, figsize=(W2, 58 * MM), sharey=True)
for k, (g, ax) in enumerate(zip(names, axes)):
    sub = pd.concat([clean, d[d.degradation == g]]); levels = list(dict.fromkeys(sub.level)); xs = np.arange(len(levels))
    for m in ("tames", "tames_quality_off", "cnn_image_only", "late_fusion_cnn_xgb"):
        s = sub[sub.model == m].set_index("level").loc[levels]
        band(ax, m, xs, s.auc_ci_low.values, s.auc_ci_high.values); line(ax, m, xs, s.auc.values, marker="o", ms=2.5)
    ax.set_xticks(xs); ax.set_xticklabels(["clean"] + [f"{l:g}" for l in levels[1:]]); ax.set_xlabel(names[g]); ax.set_ylim(0.55, 0.9)
    panel(ax, "abcd"[k], dx=-0.22 if k == 0 else -0.08)
    if k == 0:
        ax.set_ylabel("test AUC")
        ax2 = ax.twinx(); q = sub[sub.model == "tames"].set_index("level").loc[levels].mean_q.values
        ax2.plot(xs, q, color=INK2, lw=0.6, marker="s", ms=2, ls="-", alpha=0.8); ax2.set_ylim(0, 1); ax2.set_ylabel("mean $\\bar{q}$ (grey)", color=INK2, fontsize=7)
        ax2.tick_params(axis="y", labelsize=6, colors=INK2); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(INK2); ax2.grid(False)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.04), handlelength=2.4)
fig.tight_layout(w_pad=1.2, rect=(0, 0.04, 1, 1))
save(fig, "fig5_robustness")
