#!/usr/bin/env python
"""Figure 8 — (a) mean |SHAP| of the ensemble for the top-12 features (image embedding as one bar), split into the
per-learner parts ω_k·mean|φ_k| rescaled to the ensemble value, with the top-10 stability printed at the bar end;
(b) Grad-CAM gallery: 2 TP, 2 TN, 2 FP, 2 FN test cases at t* (seed 0), image | overlay, computed with the saved seed-0
model and the GradCAM class of scripts/07_explain.py (no training). Inputs: explain_tames_seed0/{shap_summary.csv,
shap_stability.json}, results.json, p_pooled_test.npy, X_test.npy, cnn_tames_seed0/model.pt.
    python scripts/plots/fig8_explanations.py"""
import sys, json; sys.path.insert(0, "scripts/plots"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from importlib import import_module
from _style import setup, save, results, panel, W2, MM, INK, INK2
from tames.config import load_config
from tames.pipeline import load_processed, tabular_block, VARIANTS
from tames.models import get_device
from tames.models.cnn import load_model

setup()
R, order, d = results(); P = np.load(f"{d}/p_pooled_test.npy")[order.index("tames")]; t = R["models"]["tames"]["threshold"]["t_star"]
w = R["per_seed"][0]["weights"]
S = pd.read_csv("runs/tn3k_fold0/explain_tames_seed0/shap_summary.csv", index_col=0)
stab = json.load(open("runs/tn3k_fold0/explain_tames_seed0/shap_stability.json"))["in_top10_frequency"]
NICE = {"image embedding (Σ|φ| over h_img)": "image embedding h_img (128 dims)", "solidity": "solidity", "irregularity": "irregularity", "eccentricity": "eccentricity",
        "taller_than_wide": "taller-than-wide", "area_frac": "area fraction", "Q_img": "Q_img", "echo_std": "echo SD", "bright_frac": "bright-spot fraction",
        "echo_mean": "echo mean", "n_components": "component count", "echo_entropy": "echo entropy", "echo_ratio": "echo ratio", "Q_struct": "Q_struct"}
top = S["ensemble"].sort_values(ascending=False).head(12)
fig = plt.figure(figsize=(W2, 92 * MM)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.3, right=0.89)
a = fig.add_subplot(gs[0]); a.grid(False); a.grid(True, axis="x", color="#e4e4e4", lw=0.5)
shades = {"dl": "#08306b", "rf": "#2171b5", "xgb": "#6baed6", "fsvm": "#c6dbef"}
ys = np.arange(len(top))[::-1]
for j, (feat, val) in enumerate(top.items()):
    parts = np.array([w[k] * S.loc[feat, k] for k in shades]); parts = parts / parts.sum() * val if parts.sum() > 0 else parts
    left = 0.0
    for (k, col), pv in zip(shades.items(), parts):
        a.barh(ys[j], pv, left=left, color=col, height=0.72, label={"dl": "DL", "rf": "RF", "xgb": "XGB", "fsvm": "FSVM"}[k] if j == 0 else None); left += pv
    if feat in stab: a.text(val + top.max() * 0.015, ys[j], f"{100 * stab[feat]:.0f} %", va="center", fontsize=6, color=INK2)
a.set_yticks(ys); a.set_yticklabels([NICE.get(f, f) for f in top.index], fontsize=6.5); a.set_xlabel("mean |SHAP| (probability units), ensemble attribution")
a.set_xlim(0, top.max() * 1.22); a.legend(loc="lower right", fontsize=6, title="learner share (ω_k·|φ_k|)", title_fontsize=6); panel(a, "a", dx=-0.62)
a.text(1.0, -0.2, "% = share of 2,000 resamples in the top ten", transform=a.transAxes, ha="right", fontsize=6, color=INK2)
# (b) Grad-CAM gallery
cfg = load_config("configs/default.yaml"); data = load_processed(cfg); tab = tabular_block(data); dev = get_device("cpu")
chans = VARIANTS["tames"]; X = np.asarray(data["X"]["test"][:, chans], dtype=np.float32); y = data["y"]["test"]
model = load_model(cfg, len(chans), "runs/tn3k_fold0/cnn_tames_seed0/model.pt", dev, tab["test"]["tab"].shape[1]).eval()
GradCAM = import_module("07_explain").GradCAM; cam = GradCAM(model)
yhat = (P >= t).astype(int); rng = np.random.default_rng(0)
pick = lambda cond: list(rng.choice(np.where(cond)[0], 2, replace=False))
cases = [("TP", pick((y == 1) & (yhat == 1))), ("TN", pick((y == 0) & (yhat == 0))), ("FP", pick((y == 0) & (yhat == 1))), ("FN", pick((y == 1) & (yhat == 0)))]
sub = gs[1].subgridspec(4, 4, wspace=0.06, hspace=0.42)
first = None
# layout: rows 0-1 = TP (cols 0-1) and TN (cols 2-3); rows 2-3 = FP and FN; each case = image | overlay
for g, (kind, idx) in enumerate(cases):
    r0, c0 = (0, 0) if kind == "TP" else (0, 2) if kind == "TN" else (2, 0) if kind == "FP" else (2, 2)
    for r, i in enumerate(idx):
        x = torch.from_numpy(X[i: i + 1]); tt = torch.from_numpy(tab["test"]["tab"][i: i + 1].astype(np.float32))
        p_dl, heat = cam(x, tt); heat = heat[0].numpy()
        ax1 = fig.add_subplot(sub[r0 + r, c0]); ax2 = fig.add_subplot(sub[r0 + r, c0 + 1])
        if first is None: first = ax1
        ax1.imshow(X[i, 0], cmap="gray", vmin=0, vmax=1); ax2.imshow(X[i, 0], cmap="gray", vmin=0, vmax=1); last = ax2.imshow(heat, cmap="magma", vmin=0, vmax=1, alpha=0.5)
        for ax in (ax1, ax2): ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); [sp_.set_visible(False) for sp_ in ax.spines.values()]
        ax1.text(1.03, -0.06, f"{kind}: y = {int(y[i])}, p = {P[i]:.2f}", transform=ax1.transAxes, ha="center", va="top", fontsize=6, color=INK)
        if r == 0: ax1.set_title(f"{kind}   image | Grad-CAM", fontsize=6.5, color=INK, pad=2, loc="left")
cax = fig.add_axes([0.905, 0.2, 0.012, 0.6]); cb = fig.colorbar(last, cax=cax); cb.set_label("Grad-CAM saliency (normalized)", fontsize=6.5); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
first.text(-0.12, 1.25, "(b)", transform=first.transAxes, fontsize=9, fontweight="bold", color=INK)
save(fig, "fig8_explanations")
