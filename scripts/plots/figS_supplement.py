#!/usr/bin/env python
"""Supplementary figures: S1 per-seed ROC spread of TAMES AI (per-seed ensemble recomputed from the saved learner
probabilities and the saved per-seed weights; pooled curve in bold); S2 confusion matrices of TAMES AI at t* and at the
Youden threshold on the test set.    python scripts/plots/figS_supplement.py"""
import sys; sys.path.insert(0, "scripts/plots")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from _style import setup, save, results, panel, W2, W1, MM, MODEL, INK, INK2

setup()
R, order, d = results(); P = np.load(f"{d}/p_pooled_test.npy")[order.index("tames")]
y = pd.read_csv("data/processed/meta_test.csv")["label"].values.astype(int)
fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 80 * MM))
for ps in R["per_seed"]:
    s = ps["seed"]; w = ps["weights"]
    p = {"dl": np.load(f"runs/tn3k_fold0/cnn_tames_seed{s}/p_test.npy"), "rf": np.load(f"runs/tn3k_fold0/learners_tames_seed{s}/p_rf_test.npy"),
         "xgb": np.load(f"runs/tn3k_fold0/learners_tames_seed{s}/p_xgb_test.npy"), "fsvm": np.load(f"runs/tn3k_fold0/learners_tames_seed{s}/p_fsvm_test.npy")}
    pe = sum(w[k] * p[k] for k in w); fpr, tpr, _ = roc_curve(y, pe)
    a.plot(fpr, tpr, color=MODEL["tames"][1], lw=0.6, alpha=0.5, label=f"seed {s}: AUC {roc_auc_score(y, pe):.3f}")
fpr, tpr, _ = roc_curve(y, P); a.plot(fpr, tpr, color=MODEL["tames"][1], lw=1.8, label=f"pooled: AUC {R['models']['tames']['test']['auc']:.3f}")
a.plot([0, 1], [0, 1], color=INK2, lw=0.5, alpha=0.7); a.set_aspect("equal"); a.set_xlim(0, 1); a.set_ylim(0, 1)
a.set_xlabel("1 − specificity"); a.set_ylabel("sensitivity"); a.legend(loc="lower right", fontsize=6); panel(a, "a")
b.grid(False); b.set_axis_off()
thr = R["models"]["tames"]["threshold"]
for k, (name, tv) in enumerate((("t* (sensitivity ≥ 0.95 on validation)", thr["t_star"]), ("Youden threshold", thr["t_youden"]))):
    cm = confusion_matrix(y, (P >= tv).astype(int), labels=[1, 0])
    ax = b.inset_axes([0.08 + 0.5 * k, 0.18, 0.4, 0.62]); ax.grid(False)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.6)
    for i in range(2):
        for j in range(2): ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred. malignant", "pred. benign"], fontsize=6); ax.set_yticks([0, 1]); ax.set_yticklabels(["malignant", "benign"] if k == 0 else ["", ""], fontsize=6)
    ax.set_title(f"{name}\nt = {tv:.3f}", fontsize=6.5, color=INK, pad=4); ax.tick_params(length=0)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
b.text(0.0, 1.0, "(b)", transform=b.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")
fig.tight_layout(w_pad=2)
save(fig, "figS1_seed_roc_confusion")
