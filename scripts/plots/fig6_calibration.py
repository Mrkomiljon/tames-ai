#!/usr/bin/env python
"""Figure 6 — calibration curves (10 quantile bins) on the test set for TAMES AI, the image-only CNN and the late
fusion, from results.json → calibration_test.bins (no recomputation).   python scripts/plots/fig6_calibration.py [run_dir]"""
import sys, json; sys.path.insert(0, "scripts/plots")
import matplotlib.pyplot as plt
from _style import setup, save, SLOT, LABEL, INK2

setup()
run = sys.argv[1] if len(sys.argv) > 1 else "runs/tn3k_fold0"
R = json.load(open(f"{run}/ensemble_tames_seed0/results.json"))["models"]
fig, ax = plt.subplots(figsize=(5.4, 5.2))
ax.plot([0, 1], [0, 1], color=INK2, lw=0.7, alpha=0.6, label="perfect calibration")
for m, mk in (("tames", "o"), ("cnn_image_only", "s"), ("late_fusion_cnn_xgb", "^")):
    c = R[m]["calibration_test"]; b = c["bins"]
    x = [q["mean_pred"] for q in b]; yv = [q["obs_rate"] for q in b]
    ax.plot(x, yv, "-", lw=1.6, color=SLOT[m], zorder=2)
    ax.plot(x, yv, mk, ms=7, color=SLOT[m], mec="white", mew=1.2, zorder=3,
            label=f"{LABEL[m]}  slope {c['slope']:.2f}, intercept {c['intercept']:.2f}, Brier {c['brier']:.3f}")
ax.set_xlabel("mean predicted probability of malignancy (10 quantile bins)"); ax.set_ylabel("observed malignant fraction")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=7.6)
save(fig, "fig6_calibration")
