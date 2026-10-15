#!/usr/bin/env python
"""Figure 1 — image flow: TN3K release -> official fold-0 train / validation and the official test set.
Counts are read from data/processed/splits.csv.   python scripts/plots/fig1_flow.py"""
import sys; sys.path.insert(0, "scripts/plots")
import pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _style import setup, save, INK, INK2, SLOT

setup()
sp = pd.read_csv("data/processed/splits.csv")
n = sp.groupby("split").label.agg(["size", "sum"])
tv_n, tv_m = int(n.loc[["train", "val"], "size"].sum()), int(n.loc[["train", "val"], "sum"].sum())
fig, ax = plt.subplots(figsize=(8.6, 4.4)); ax.set_axis_off(); ax.grid(False); ax.set_xlim(0, 12); ax.set_ylim(0, 6)

def box(x, y, w, h, title, lines, accent=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white",
                                ec=accent or INK2, lw=1.4 if accent else 0.9))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=9, color=INK, weight="bold")
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.72 - 0.34 * i, t, ha="center", va="center", fontsize=7.8, color=INK2)

def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=INK2, lw=0.9))

box(3.0, 4.55, 6.0, 1.25, f"TN3K release: {len(sp):,} B-mode images",
    ["one radiologist mask and one benign/malignant label per image", "no patient identifiers; no image excluded"])
box(0.3, 2.55, 5.4, 1.25, f"Train + validation files: {tv_n:,} images", [f"{tv_m:,} malignant ({100 * tv_m / tv_n:.1f} %)", "official fold-0 lists"])
box(6.3, 2.55, 5.4, 1.25, f"Official test set: {n.loc['test', 'size']:,} images",
    [f"{n.loc['test', 'sum']:,} malignant ({100 * n.loc['test', 'sum'] / n.loc['test', 'size']:.1f} %)", "scored once, after every choice was frozen"], accent=SLOT["tames"])
box(0.3, 0.35, 2.6, 1.45, f"Training: {n.loc['train', 'size']:,}", [f"{n.loc['train', 'sum']:,} malignant", "fits f_θ, learners,", "descriptor statistics"])
box(3.1, 0.35, 2.6, 1.45, f"Validation: {n.loc['val', 'size']:,}", [f"{n.loc['val', 'sum']:,} malignant", "selects α β γ λ, τ, ω,", "hyperparameters, t*"])
arrow(5.2, 4.55, 3.0, 3.82); arrow(6.8, 4.55, 9.0, 3.82); arrow(2.0, 2.55, 1.6, 1.82); arrow(3.9, 2.55, 4.4, 1.82)
ax.text(9.0, 0.95, "image-level split\n(TN3K carries no patient IDs)", ha="center", va="center", fontsize=8, color=INK2, style="italic")
save(fig, "fig1_flow")
