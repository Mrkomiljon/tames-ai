#!/usr/bin/env python
"""Figure 3 — real TN3K test images through the fuzzy front end. Six rows (3 benign on top, 3 malignant below) chosen
deterministically (seed 0) so that one false positive and one false negative at t* are included; columns: raw image with
radiologist contour and the 1.5× crop box, crop x̃, μ, F, Otsu(F) over the crop with the contour and the per-image Dice.
Inputs: data/raw (images, masks), data/processed/{splits,meta_test}.csv, X_test.npy, fuzzy_params.json, saved predictions.
    python scripts/plots/fig3_samples.py"""
import sys, json; sys.path.insert(0, "scripts/plots"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from _style import setup, save, results, W2, MM, INK, INK2
from tames.fuzzy import fuzzy_maps, otsu_binarize
from tames.data import load_image, load_mask

setup()
R, order, d = results(); P = np.load(f"{d}/p_pooled_test.npy")[order.index("tames")]; t = R["models"]["tames"]["threshold"]["t_star"]
meta = pd.read_csv("data/processed/meta_test.csv"); sp = pd.read_csv("data/processed/splits.csv").set_index("id")
X = np.load("data/processed/X_test.npy", mmap_mode="r"); fz = json.load(open("data/processed/fuzzy_params.json"))
y = meta.label.values; yhat = (P >= t).astype(int); rng = np.random.default_rng(0)
pick = lambda cond, k: list(rng.choice(np.where(cond)[0], k, replace=False))
rows = pick((y == 0) & (yhat == 0), 2) + pick((y == 0) & (yhat == 1), 1) + pick((y == 1) & (yhat == 1), 2) + pick((y == 1) & (yhat == 0), 1)
print("rows:", [(meta.id[i], int(y[i]), round(float(P[i]), 2)) for i in rows])

fig, axes = plt.subplots(6, 5, figsize=(W2, 6 * 31 * MM + 8 * MM), gridspec_kw={"wspace": 0.04, "hspace": 0.16})
red = ListedColormap([(0, 0, 0, 0), (0.84, 0.37, 0.0, 0.45)])  # Okabe–Ito vermilion overlay
for r, i in enumerate(rows):
    m = meta.iloc[i]; row = sp.loc[m.id]; img = load_image(row.image_path); mask = load_mask(row.mask_path) > 0
    y0, y1, x0, x1 = map(int, m.crop.split(","))
    maps = fuzzy_maps(img, fz, fz["snr0"]); seg = otsu_binarize(maps["F"], fz["polarity"])
    crop = lambda a: a[y0:y1, x0:x1]
    x_t = np.asarray(X[i], dtype=np.float32)
    ax = axes[r, 0]; ax.imshow(img, cmap="gray", vmin=0, vmax=255); ax.contour(mask, levels=[0.5], colors=["#D55E00"], linewidths=0.6)
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#0072B2", lw=0.7, ls="--"))
    axes[r, 1].imshow(x_t[0], cmap="gray", vmin=0, vmax=1)
    im_mu = axes[r, 2].imshow(x_t[1], cmap="viridis", vmin=0, vmax=1); im_F = axes[r, 3].imshow(x_t[2], cmap="viridis", vmin=0, vmax=1)
    ax = axes[r, 4]; ax.imshow(crop(img), cmap="gray", vmin=0, vmax=255); ax.imshow(crop(seg).astype(int), cmap=red, vmin=0, vmax=1, interpolation="nearest")
    ax.contour(crop(mask), levels=[0.5], colors=["#0072B2"], linewidths=0.6)
    ax.text(0.03, 0.04, f"Dice {m.dice_afmo:.2f}", transform=ax.transAxes, fontsize=6, color="white", va="bottom", bbox=dict(fc="black", alpha=0.5, lw=0, pad=1.5))
    axes[r, 0].set_ylabel(f"{m.id}\n{'malignant' if y[i] else 'benign'} · p = {P[i]:.2f}\n{'(FN)' if (y[i] and not yhat[i]) else '(FP)' if (not y[i] and yhat[i]) else ''}", fontsize=6.5, color=INK, rotation=0, ha="right", va="center", labelpad=28)
    for ax in axes[r]: ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
for c, ttl in enumerate(["(i) image, delineation, crop box", "(ii) crop x̃", "(iii) membership μ", "(iv) map F", "(v) Otsu(F) and delineation"]):
    axes[0, c].set_title(ttl, fontsize=7, color=INK, pad=3)
for c, im in ((2, im_mu), (3, im_F)):
    cb = fig.colorbar(im, ax=axes[:, c].tolist(), orientation="horizontal", fraction=0.025, pad=0.012, aspect=18, shrink=0.72); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
save(fig, "fig3_samples")
