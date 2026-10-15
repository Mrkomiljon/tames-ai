#!/usr/bin/env python
"""Figure 1 — STARD-style image flow of TN3K through the official fold-0 split (counts from data/processed/splits.csv).
    python scripts/plots/fig1_flow.py"""
import sys; sys.path.insert(0, "scripts/plots")
import pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _style import setup, save, W1, MM, INK, INK2, TINT

setup()
sp = pd.read_csv("data/processed/splits.csv"); n = sp.groupby("split").label.agg(["size", "sum"])
tv_n, tv_m = int(n.loc[["train", "val"], "size"].sum()), int(n.loc[["train", "val"], "sum"].sum())
fig, ax = plt.subplots(figsize=(W1, 72 * MM)); ax.set_axis_off(); ax.grid(False); ax.set_xlim(0, 85); ax.set_ylim(0, 72)
T, S = 6.6, 5.6  # title / line font sizes (pt)

def box(x, y, w, h, title, lines, tint=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=TINT if tint else "white", ec=INK2, lw=0.6))
    ax.text(x + w / 2, y + h - 2.9, title, ha="center", va="center", fontsize=T, fontweight="bold", color=INK)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.3 - 3.1 * i, t, ha="center", va="center", fontsize=S, color=INK2)

def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, color=INK2, lw=0.6, shrinkA=0, shrinkB=0))

box(16, 59, 53, 12, f"TN3K release: {len(sp):,} images", ["one radiologist mask + one label per image", "no patient identifiers; none excluded"])
box(2, 38, 39, 14.5, f"Train + validation: {tv_n:,}", [f"{tv_m:,} malignant ({100 * tv_m / tv_n:.1f} %)", "official fold-0 lists", "all tuning decisions"])
box(44, 38, 39, 14.5, f"Official test set: {n.loc['test', 'size']:,}", [f"{n.loc['test', 'sum']:,} malignant ({100 * n.loc['test', 'sum'] / n.loc['test', 'size']:.1f} %)", "scored once, after every", "choice was frozen"], tint=True)
box(2, 13, 25.5, 17.5, f"Training: {n.loc['train', 'size']:,}", [f"{n.loc['train', 'sum']:,} malignant", "fits f_θ, learners,", "descriptor statistics"])
box(30, 13, 25.5, 17.5, f"Validation: {n.loc['val', 'size']:,}", [f"{n.loc['val', 'sum']:,} malignant", "selects α, β, γ, λ, τ, ω,", "learner settings, t*"])
arrow(34, 59, 21.5, 52.5); arrow(51, 59, 63.5, 52.5); arrow(16, 38, 14.75, 30.5); arrow(27, 38, 42.75, 30.5)
ax.text(70, 21, "image-level split:\nTN3K carries no\npatient identifiers", ha="center", va="center", fontsize=S, color=INK2, style="italic")
save(fig, "fig1_flow")
