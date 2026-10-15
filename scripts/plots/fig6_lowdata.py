#!/usr/bin/env python
"""Figure 6 — test AUC vs fraction of training images (runs/lowdata_table.csv), 95 % bootstrap CI as error bars.
    python scripts/plots/fig6_lowdata.py"""
import sys; sys.path.insert(0, "scripts/plots")
import pandas as pd, matplotlib.pyplot as plt
from _style import setup, save, W1, MM, MODEL

setup()
L = pd.read_csv("runs/lowdata_table.csv")
fig, ax = plt.subplots(figsize=(W1, 62 * MM))
for i, m in enumerate(["tames", "cnn_image_only", "late_fusion_cnn_xgb", "xgb_struct_only"]):
    s = L[L.model == m].sort_values("train_pct"); lab, c, ls, lw = MODEL[m]
    x = s.train_pct * (1 + 0.025 * (i - 1.5))  # small horizontal offset so error bars do not overlap
    ax.errorbar(x, s.auc, yerr=[s.auc - s.ci_low, s.ci_high - s.auc], fmt="o", ms=3, color=c, ecolor=c, elinewidth=0.7, capsize=1.5, zorder=3)
    ax.plot(x, s.auc, ls, color=c, lw=lw, label=lab, zorder=2)
ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100]); ax.set_xticklabels(["10 %", "25 %", "50 %", "100 %"]); ax.minorticks_off()
ax.set_xlabel("fraction of training images"); ax.set_ylabel("test AUC"); ax.set_ylim(0.55, 0.88)
ax.legend(loc="lower right", fontsize=6.3, frameon=False)
save(fig, "fig6_lowdata")
