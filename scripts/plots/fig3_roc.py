#!/usr/bin/env python
"""Figure 3 — ROC curves of all Table-3 models on the official test set, from the pooled (seed-averaged)
predictions saved by step 05; AUC with 95 % bootstrap CI in the legend.   python scripts/plots/fig3_roc.py [run_dir]"""
import sys, json; sys.path.insert(0, "scripts/plots")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from _style import setup, save, SLOT, LABEL, ORDER, STYLE, INK2

setup()
run = sys.argv[1] if len(sys.argv) > 1 else "runs/tn3k_fold0"
d = f"{run}/ensemble_tames_seed0"
order = json.load(open(f"{d}/model_order.json")); P = np.load(f"{d}/p_pooled_test.npy")
R = json.load(open(f"{d}/results.json"))["models"]
y = pd.read_csv("data/processed/meta_test.csv")["label"].values.astype(int)
fig, ax = plt.subplots(figsize=(5.6, 5.4))
for m in ORDER:
    fpr, tpr, _ = roc_curve(y, P[order.index(m)]); ls, lw = STYLE[m]; ci = R[m]["test_ci"]["auc"]
    ax.plot(fpr, tpr, ls, lw=lw, color=SLOT[m], label=f"{LABEL[m]}  {R[m]['test']['auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
            zorder=3 if m == "tames" else 2)
ax.plot([0, 1], [0, 1], color=INK2, lw=0.7, ls="-", alpha=0.6, zorder=1)
ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.legend(loc="lower right", fontsize=7.6, title=f"test set, n = {len(y)} · AUC [95 % CI]", title_fontsize=7.8)
save(fig, "fig3_roc")
