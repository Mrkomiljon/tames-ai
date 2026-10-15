#!/usr/bin/env python
"""Figure 8 — (a) mean |SHAP| of the ensemble for the top-12 features, x-axis scaled to the descriptors; the image-embedding
bar is hatched and clipped at the axis edge with its value printed; bars are split into the per-learner parts ω_k·mean|φ_k|
rescaled to the ensemble value; top-10 stability printed right of each bar. (b) Grad-CAM gallery, rows TP / TN / FP / FN,
two cases per row, image | overlay (overlay alpha proportional to saliency), from the saved seed-0 backbone and the GradCAM
class of scripts/07_explain.py (inference only).    python scripts/plots/fig8_explanations.py"""
import sys, json; sys.path.insert(0, "scripts/plots"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from matplotlib import cm as mcm
from importlib import import_module
from _style import setup, save, results, W2, MM, INK, INK2
from tames.config import load_config
from tames.pipeline import load_processed, tabular_block, VARIANTS
from tames.models import get_device
from tames.models.cnn import load_model

setup()
R, order, d = results(); P = np.load(f"{d}/p_pooled_test.npy")[order.index("tames")]; t = R["models"]["tames"]["threshold"]["t_star"]
w = R["per_seed"][0]["weights"]
S = pd.read_csv("runs/tn3k_fold0/explain_tames_seed0/shap_summary.csv", index_col=0)
stab = json.load(open("runs/tn3k_fold0/explain_tames_seed0/shap_stability.json"))["in_top10_frequency"]
NICE = {"image embedding (Σ|φ| over h_img)": "image embedding $h_{img}$", "solidity": "solidity", "irregularity": "irregularity", "eccentricity": "eccentricity",
        "taller_than_wide": "taller-than-wide", "area_frac": "area fraction", "Q_img": "$Q_{img}$", "echo_std": "echo SD", "bright_frac": "bright-spot fraction",
        "echo_mean": "echo mean", "n_components": "component count", "echo_entropy": "echo entropy", "echo_ratio": "echo ratio", "Q_struct": "$Q_{struct}$"}
top = S["ensemble"].sort_values(ascending=False).head(12); emb = [f for f in top.index if f.startswith("image")][0]
xmax = 1.1 * top.drop(emb).max(); total = S["ensemble"].sum()
HFIG = 4 * 28 * MM + 14 * MM
fig = plt.figure(figsize=(W2, HFIG)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.3], wspace=0.14, left=0.14, right=0.93, top=0.95, bottom=0.05)
a = fig.add_subplot(gs[0]); a.grid(False); a.grid(True, axis="x", color="#e4e4e4", lw=0.5)
shades = {"dl": "#08306b", "rf": "#2171b5", "xgb": "#6baed6", "fsvm": "#c6dbef"}
ys = np.arange(len(top))[::-1]
for j, (feat, val) in enumerate(top.items()):
    if feat == emb:
        a.barh(ys[j], xmax, color="white", edgecolor="#0072B2", hatch="////", height=0.72, lw=0.6, clip_on=True)
        a.annotate("", xy=(xmax * 0.82, ys[j]), xytext=(xmax * 0.68, ys[j]), arrowprops=dict(arrowstyle="-|>", color="#0072B2", lw=0.8, mutation_scale=7))
        a.text(xmax * 0.02, ys[j] + 0.5, f"{val:.3f} ({100 * val / total:.1f} % of total), not to scale", fontsize=5.6, color="#0072B2", va="bottom")
    else:
        parts = np.array([w[k] * S.loc[feat, k] for k in shades]); parts = parts / parts.sum() * val if parts.sum() > 0 else parts
        left = 0.0
        for (k, col), pv in zip(shades.items(), parts):
            a.barh(ys[j], pv, left=left, color=col, height=0.72, label={"dl": "DL", "rf": "RF", "xgb": "XGB", "fsvm": "FSVM"}[k] if j == 1 else None); left += pv
    if feat in stab:
        if feat == emb: a.text(xmax * 0.97, ys[j], f"{100 * stab[feat]:.0f} %", va="center", ha="right", fontsize=6, color=INK2, bbox=dict(fc="white", ec="none", pad=1.0))
        else: a.text(val + xmax * 0.015, ys[j], f"{100 * stab[feat]:.0f} %", va="center", fontsize=6, color=INK2)
a.set_yticks(ys); a.set_yticklabels([NICE.get(f, f) for f in top.index], fontsize=6.5); a.set_xlim(0, xmax); a.set_ylim(-0.6, len(top) - 0.3)
a.set_xlabel("mean |SHAP| (probability units), ensemble attribution"); a.tick_params(axis="x", labelsize=6)
a.legend(loc="lower right", fontsize=5.8, title="learner share $\\omega_k \\cdot |\\phi_k|$", title_fontsize=5.8, bbox_to_anchor=(1.0, 0.0))
a.text(0.0, -0.1, "% = share of 2,000 resamples in the top ten", transform=a.transAxes, ha="left", fontsize=5.8, color=INK2)
fig.text(0.015, 0.965, "(a)", fontsize=9, fontweight="bold", color=INK, va="bottom")
# (b) Grad-CAM gallery
cfg = load_config("configs/default.yaml"); data = load_processed(cfg); tab = tabular_block(data); dev = get_device("cpu")
chans = VARIANTS["tames"]; X = np.asarray(data["X"]["test"][:, chans], dtype=np.float32); y = data["y"]["test"]
model = load_model(cfg, len(chans), "runs/tn3k_fold0/cnn_tames_seed0/model.pt", dev, tab["test"]["tab"].shape[1]).eval()
GradCAM = import_module("07_explain").GradCAM; cam = GradCAM(model)
yhat = (P >= t).astype(int); rng = np.random.default_rng(0)
pick = lambda cond: list(rng.choice(np.where(cond)[0], 2, replace=False))
cases = [("TP", pick((y == 1) & (yhat == 1))), ("TN", pick((y == 0) & (yhat == 0))), ("FP", pick((y == 0) & (yhat == 1))), ("FN", pick((y == 1) & (yhat == 0)))]
sub = gs[1].subgridspec(4, 5, width_ratios=[1, 1, 1, 1, 0.07], wspace=0.04, hspace=0.3)
first = None; cmap = plt.get_cmap("magma")
for r, (kind_, idx) in enumerate(cases):
    for c, i in enumerate(idx):
        x = torch.from_numpy(X[i: i + 1]); tt = torch.from_numpy(tab["test"]["tab"][i: i + 1].astype(np.float32))
        p_dl, heat = cam(x, tt); heat = heat[0].numpy()
        ax1 = fig.add_subplot(sub[r, 2 * c]); ax2 = fig.add_subplot(sub[r, 2 * c + 1])
        if first is None: first = ax1
        ax1.imshow(X[i, 0], cmap="gray", vmin=0, vmax=1); ax2.imshow(X[i, 0], cmap="gray", vmin=0, vmax=1)
        rgba = cmap(heat); rgba[..., 3] = 0.9 * heat; last = ax2.imshow(rgba, vmin=0, vmax=1)
        for ax in (ax1, ax2): ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); [sp_.set_visible(False) for sp_ in ax.spines.values()]
        ax1.text(1.02, -0.04, f"y = {int(y[i])}, p = {P[i]:.2f}", transform=ax1.transAxes, ha="center", va="top", fontsize=6, color=INK)
        if c == 0: ax1.set_ylabel(kind_, fontsize=7, fontweight="bold", color=INK, rotation=0, ha="right", va="center", labelpad=6)
        if r == 0: ax1.set_title("image", fontsize=6, color=INK2, pad=2); ax2.set_title("Grad-CAM", fontsize=6, color=INK2, pad=2)
cax = fig.add_subplot(sub[:, 4]); pos = cax.get_position(); cax.set_position([pos.x0 + 0.01, 0.25, pos.width, 0.5])
cb = fig.colorbar(mcm.ScalarMappable(cmap="magma"), cax=cax); cb.set_label("Grad-CAM saliency (normalized); overlay opacity $\\propto$ saliency", fontsize=6); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
fig.text(0.40, 0.965, "(b)", fontsize=9, fontweight="bold", color=INK, va="bottom")
save(fig, "fig8_explanations")
