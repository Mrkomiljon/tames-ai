#!/usr/bin/env python
"""Figure 4 — ROC curves on the official test set from the pooled (seed-averaged) saved predictions:
(a) expert delineation (runs/tn3k_fold0, Table 3), (b) fully automatic (runs/tn3k_automask, Table 5).
    python scripts/plots/fig4_roc.py"""
import sys; sys.path.insert(0, "scripts/plots")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from _style import setup, save, results, line, panel, W2, MM, MODEL, ROC_ORDER, INK2

setup()
fig, axes = plt.subplots(1, 2, figsize=(W2, 88 * MM))
for ax, letter, run, proc, note in ((axes[0], "a", "runs/tn3k_fold0", "data/processed", "expert delineation"), (axes[1], "b", "runs/tn3k_automask", "data/processed_afmo", "automatic delineation")):
    R, order, d = results(run); P = np.load(f"{d}/p_pooled_test.npy"); y = pd.read_csv(f"{proc}/meta_test.csv")["label"].values.astype(int)
    for m in ROC_ORDER:
        if m not in R["models"]: continue
        fpr, tpr, _ = roc_curve(y, P[order.index(m)]); ci = R["models"][m]["test_ci"]["auc"]
        line(ax, m, fpr, tpr, label=f"{MODEL[m][0]} — {R['models'][m]['test']['auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", zorder=3 if m == "tames" else 2)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.5, ls="-", alpha=0.7, zorder=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity")
    ax.legend(loc="lower right", fontsize=5.8, handlelength=2.4, borderaxespad=0.3); panel(ax, letter)
    ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=7, color=INK2, va="top", ha="left")
fig.tight_layout(w_pad=2)
save(fig, "fig4_roc")
