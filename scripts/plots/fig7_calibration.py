#!/usr/bin/env python
"""Figure 7 — (a) reliability curves (10 quantile bins, point area ∝ bin count) with slope/intercept in the legend;
(b) density of predicted test probabilities with t* of TAMES AI. Inputs: results.json (calibration bins, thresholds),
p_pooled_test.npy.    python scripts/plots/fig7_calibration.py"""
import sys; sys.path.insert(0, "scripts/plots")
import numpy as np, matplotlib.pyplot as plt
from _style import setup, save, results, panel, W2, MM, MODEL, INK2

setup()
R, order, d = results(); P = np.load(f"{d}/p_pooled_test.npy"); t = R["models"]["tames"]["threshold"]["t_star"]
area = lambda n: np.clip(10 + 110 * (np.asarray(n, dtype=float) - 30) / 60, 10, 120)  # pt²: n = 30 → 10, n = 90 → 120
fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 80 * MM), gridspec_kw={"width_ratios": [1, 1.15]})
a.plot([0, 1], [0, 1], color=INK2, lw=0.5, alpha=0.7)
for m, mk in (("tames", "o"), ("cnn_image_only", "s"), ("late_fusion_cnn_xgb", "^")):
    c = R["models"][m]["calibration_test"]; x = [q["mean_pred"] for q in c["bins"]]; yv = [q["obs_rate"] for q in c["bins"]]; n = np.array([q["n"] for q in c["bins"]])
    lab, col, ls, lw = MODEL[m]
    a.plot(x, yv, ls, color=col, lw=lw * 0.8, zorder=2)
    a.scatter(x, yv, s=area(n), marker=mk, color=col, edgecolor="white", linewidth=0.5, zorder=3, label=f"{lab}: slope {c['slope']:.2f}, intercept {c['intercept']:.2f}")
    b.hist(P[order.index(m)], bins=np.linspace(0, 1, 26), density=True, histtype="step", color=col, ls=ls, lw=lw, label=lab)
a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect("equal"); a.set_xlabel("mean predicted probability (bin)"); a.set_ylabel("observed malignant fraction")
leg1 = a.legend(loc="upper left", fontsize=5.8); a.add_artist(leg1)
handles = [a.scatter([], [], s=area(k), color=INK2, edgecolor="white", linewidth=0.5, label=f"n = {k}") for k in (30, 60, 90)]
a.legend(handles=handles, loc="lower right", fontsize=5.8, title="bin count", title_fontsize=5.8, labelspacing=1.1, borderpad=0.8); panel(a, "a")
b.axvline(t, color=MODEL["tames"][1], ls="--", lw=0.8, label=f"t* of TAMES AI = {t:.3f}")
b.set_xlim(0, 1); b.set_xlabel("predicted probability of malignancy"); b.set_ylabel("density (test set, n = 614)"); b.legend(loc="upper right", fontsize=6); panel(b, "b", dx=-0.12)
fig.tight_layout(w_pad=2)
save(fig, "fig7_calibration")
