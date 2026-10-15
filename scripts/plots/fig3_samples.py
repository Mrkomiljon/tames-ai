#!/usr/bin/env python
"""Figure 3 — real TN3K test images through the fuzzy front end. Six rows (3 benign on top, 3 malignant below) chosen
deterministically (seed 0) so that one false positive and one false negative at t* are included. Columns: (i) image padded
to square with the radiologist contour and the 1.5× crop box, (ii) crop x̃, (iii) μ, (iv) F (both with a per-map 1–99th
percentile display stretch, magma), (v) Otsu(F) as a solid contour and the delineation as a dashed contour over the
greyscale crop, with the per-image Dice. Inputs: data/raw (images, masks), data/processed/{splits,meta_test}.csv,
X_test.npy, fuzzy_params.json, saved predictions. Maps are recomputed from the raw image with the saved parameters.
    python scripts/plots/fig3_samples.py"""
import sys, json; sys.path.insert(0, "scripts/plots"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
from _style import setup, save, results, W2, MM, INK, INK2
from tames.fuzzy import fuzzy_maps, otsu_binarize
from tames.data import load_image, load_mask

setup()
C1, C2 = "#0072B2", "#D55E00"  # Otsu(F) contour / radiologist delineation
R, order, d = results(); P = np.load(f"{d}/p_pooled_test.npy")[order.index("tames")]; t = R["models"]["tames"]["threshold"]["t_star"]
meta = pd.read_csv("data/processed/meta_test.csv"); sp = pd.read_csv("data/processed/splits.csv").set_index("id")
X = np.load("data/processed/X_test.npy", mmap_mode="r"); fz = json.load(open("data/processed/fuzzy_params.json"))
y = meta.label.values; yhat = (P >= t).astype(int); rng = np.random.default_rng(0)
pick = lambda cond, k: list(rng.choice(np.where(cond)[0], k, replace=False))
rows = pick((y == 0) & (yhat == 0), 2) + pick((y == 0) & (yhat == 1), 1) + pick((y == 1) & (yhat == 1), 2) + pick((y == 1) & (yhat == 0), 1)
kind = lambda i: "TP" if y[i] and yhat[i] else "TN" if not y[i] and not yhat[i] else "FP" if yhat[i] else "FN"
print("rows:", [(meta.id[i], kind(i), round(float(P[i]), 2)) for i in rows])

def square(a, fill=0):
    h, w = a.shape[:2]; s = max(h, w); out = np.full((s, s) + a.shape[2:], fill, dtype=a.dtype)
    oy, ox = (s - h) // 2, (s - w) // 2; out[oy:oy + h, ox:ox + w] = a; return out, oy, ox

def stretch(m):
    lo, hi = np.percentile(m, [1, 99]); return np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1)

PW = 27 * MM
fig = plt.figure(figsize=(W2, 6 * PW + 10 * MM))
gs = fig.add_gridspec(6, 6, width_ratios=[1, 1, 1, 1, 1, 0.05], wspace=0.05, hspace=0.06, left=0.17, right=0.985, top=0.955, bottom=0.01)
axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(6)])
for r, i in enumerate(rows):
    m = meta.iloc[i]; row = sp.loc[m.id]; img = load_image(row.image_path); mask = load_mask(row.mask_path) > 0
    y0, y1, x0, x1 = map(int, m.crop.split(","))
    maps = fuzzy_maps(img, fz, fz["snr0"]); seg = otsu_binarize(maps["F"], fz["polarity"])
    x_t = np.asarray(X[i], dtype=np.float32)
    sq, oy, ox = square(img)
    ax = axes[r, 0]; ax.imshow(sq, cmap="gray", vmin=0, vmax=255)
    msq, _, _ = square(mask.astype(np.uint8)); ax.contour(msq, levels=[0.5], colors=[C2], linewidths=0.6)
    ax.add_patch(plt.Rectangle((x0 + ox, y0 + oy), x1 - x0, y1 - y0, fill=False, ec=C1, lw=0.7, ls="--"))
    axes[r, 1].imshow(x_t[0], cmap="gray", vmin=0, vmax=1)
    im = axes[r, 2].imshow(stretch(x_t[1]), cmap="magma", vmin=0, vmax=1); axes[r, 3].imshow(stretch(x_t[2]), cmap="magma", vmin=0, vmax=1)
    ax = axes[r, 4]; ax.imshow(x_t[0], cmap="gray", vmin=0, vmax=1)
    cs = cv2.resize(seg[y0:y1, x0:x1].astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
    cm = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
    ax.contour(cs, levels=[0.5], colors=[C1], linewidths=0.6); ax.contour(cm, levels=[0.5], colors=[C2], linewidths=0.6, linestyles="dashed")
    ax.text(0.04, 0.04, f"Dice {m.dice_afmo:.2f}", transform=ax.transAxes, fontsize=5.8, color=INK, va="bottom", bbox=dict(fc="white", ec="none", pad=1.2))
    axes[r, 0].set_ylabel(f"{m.id} · {'malignant' if y[i] else 'benign'} · {kind(i)} · p = {P[i]:.2f}", fontsize=6, color=INK, rotation=0, ha="right", va="center", labelpad=4)
    for ax in axes[r]: ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); [s_.set_visible(False) for s_ in ax.spines.values()]
for c, ttl in enumerate(["(i) image, delineation, crop box", "(ii) crop $\\tilde{x}$", "(iii) membership $\\mu$", "(iv) map $F$", "(v) Otsu($F$) and delineation"]):
    axes[0, c].set_title(ttl, fontsize=6.8, color=INK, pad=4)
cax = fig.add_subplot(gs[:, 5]); cb = fig.colorbar(im, cax=cax); cb.set_label("normalized (display range 1–99th percentile)", fontsize=6); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
cax.set_position([cax.get_position().x0, 0.3, cax.get_position().width, 0.4])
fig.text(0.17 + (0.985 - 0.17) * 0.9, 0.975, "", fontsize=5)
save(fig, "fig3_samples")
