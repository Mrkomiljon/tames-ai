#!/usr/bin/env python
"""Figure 2 — computational architecture of TAMES AI (170 mm, double column).

(a) the full model as a data-flow graph with tensor shapes: image branch (fuzzy front end →
    two-channel ResNet-18 → 128-d embedding), structured branch, quality-weighted fusion,
    four learners, adaptive ensemble and decision rule;
(b) the fuzzy front end, Eqs. (1)–(8);
(c) the residual block of the backbone.
Solid = evaluated on TN3K, dashed = defined by the framework but without data in TN3K,
blue = contribution of this work. Equation numbers refer to Section 3.
    python scripts/plots/fig2_architecture.py [--out paper/figures]
"""
import argparse
import sys
from pathlib import Path

from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle_diagram import (ACCENT, ACCENT_FILL, GREY, INK, RULE, TINT, arrow, box, canvas,  # noqa: E402
                              label, save, setup_font)

ROOT = Path(__file__).resolve().parents[2]
CONV_FILL, CONV_EDGE = "#fdf1e7", "#d55e00"     # convolutional blocks (Okabe–Ito vermillion)
FC_FILL, FC_EDGE = "#e8f4ee", "#009e73"         # dense layers (Okabe–Ito green)
PANEL = "#f8f8f6"
T, L, S = 6.1, 5.1, 4.6                          # title / line / shape-label sizes (pt)
DASH = (0, (2.2, 1.6))


def node(ax, x, y, text, r=2.1, size=7.0):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=INK, linewidth=0.8, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, zorder=4)


def stack(ax, x, y, w, h, n=2, fill=CONV_FILL, edge=CONV_EDGE, off=0.9):
    """n offset rounded boxes = repeated blocks."""
    for i in reversed(range(n)):
        ax.add_patch(FancyBboxPatch((x + i * off, y - i * off), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor=fill, edgecolor=edge, linewidth=0.8, zorder=2 + (n - i) * 0.01))


def shape(ax, x, y, text, size=S, color=GREY):
    ax.text(x, y, text, ha="center", va="top", fontsize=size, color=color, zorder=3)


def panel_title(ax, x, y, letter, text):
    ax.text(x, y, letter, ha="left", va="center", fontsize=8, fontweight="bold", color=INK, zorder=3)
    ax.text(x + 4.5, y, text, ha="left", va="center", fontsize=6.8, color=INK, zorder=3)


def dashed_arrow(ax, p0, p1):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=RULE, lw=0.8, ls=DASH, zorder=1)
    ax.add_patch(FancyArrowPatch((p1[0] - (p1[0] - p0[0]) * 1e-3, p1[1] - (p1[1] - p0[1]) * 1e-3), p1,
                                 arrowstyle="-|>,head_length=3,head_width=1.75", color=RULE, linewidth=0.8,
                                 mutation_scale=1, shrinkA=0, shrinkB=0, zorder=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    a = ap.parse_args()
    setup_font()
    W, H = 170, 142
    fig, ax = canvas(W, H)

    # ------------------------------------------------------------------ (a) image branch, lane 1
    panel_title(ax, 2, 2, "(a)", "TAMES AI as a data-flow graph; shapes are per image")
    yc = 20                                                   # lane-1 centre line
    box(ax, 2, yc - 8, 13, 16, "Input $I_n$", ["B-mode crop", "224 × 224 × 1"], fill=TINT,
        title_size=T, line_size=L)
    box(ax, 17, yc - 11, 24, 22, "Fuzzy front end", ["ATCFM $\\mu(p)$ — Eq. (4)", "AFMO $F$ — Eq. (8)",
                                                  "quality $q$ — Eq. (3)", "panel (b)"],
        fill=ACCENT_FILL, edge=ACCENT, lw=1.1, title_color=ACCENT, title_size=T, line_size=L)
    arrow(ax, (15, yc), (17, yc))
    box(ax, 44, yc - 8, 11, 16, "$[\\tilde x, F]$", ["2 channels", "224 × 224 × 2"], title_size=T, line_size=L)
    arrow(ax, (41, yc), (44, yc))

    # ResNet-18 container
    rx0, rx1 = 58, 115
    ax.add_patch(FancyBboxPatch((rx0, 6), rx1 - rx0, 30, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=PANEL, edgecolor=RULE, linewidth=0.7, zorder=1))
    ax.text(rx0 + 2, 8.6, "ResNet-18 backbone $f_\\theta$ — Eq. (9)", ha="left", va="center", fontsize=T,
            fontweight="bold", color=INK, zorder=3)
    ax.text(rx0 + 2, 33.3, "ImageNet-1k initialization; stem widened to two channels", ha="left", va="center",
            fontsize=S, color=GREY, zorder=3)
    arrow(ax, (55, yc), (rx0 + 2.5, yc))
    # stem
    sx = rx0 + 2.5
    ax.add_patch(FancyBboxPatch((sx, yc - 10), 7, 20, boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor=CONV_FILL, edgecolor=CONV_EDGE, linewidth=0.8, zorder=2))
    ax.text(sx + 3.5, yc, "7×7 conv\n64, /2\nmaxpool /2", ha="center", va="center", fontsize=S, color=INK, zorder=3)
    shape(ax, sx + 3.5, yc + 10.8, "56² × 64")
    # four stages, heights follow the spatial size
    stages = [("64", 56, 18), ("128", 28, 14.5), ("256", 14, 11), ("512", 7, 8)]
    x = sx + 7 + 3
    cx_last = None
    for ch, sp, h in stages:
        stack(ax, x, yc - h / 2, 7, h)
        ax.text(x + 3.9, yc - 0.4, f"basic\nblock\n×2", ha="center", va="center", fontsize=S, color=INK, zorder=4)
        shape(ax, x + 4, yc + h / 2 + 1.3, f"{sp}² × {ch}")
        arrow(ax, (x - 3, yc), (x, yc))
        cx_last = x + 4
        x += 7 + 3.4
    # GAP
    ax.add_patch(FancyBboxPatch((x, yc - 4), 6, 8, boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor=FC_FILL, edgecolor=FC_EDGE, linewidth=0.8, zorder=2))
    ax.text(x + 3, yc, "GAP", ha="center", va="center", fontsize=S, color=INK, zorder=3)
    shape(ax, x + 3, yc + 5.3, "512")
    arrow(ax, (x - 2.5, yc), (x, yc))
    ax.text(cx_last, 29.6, "Grad-CAM tap", ha="center", va="center", fontsize=S, color=GREY,
            fontstyle="italic", zorder=3)

    # FC embedding
    fx = rx1 + 3
    box(ax, fx, yc - 7, 16, 14, "FC 512 $\\rightarrow$ 128", ["ReLU, dropout 0.3", "$h_{img} \\in \\mathbb{R}^{128}$"],
        fill=FC_FILL, edge=FC_EDGE, title_size=T, line_size=L)
    arrow(ax, (rx1, yc), (fx, yc))
    # × Q_img
    mx = fx + 16 + 6.5
    node(ax, mx, yc, "×")
    arrow(ax, (fx + 16, yc), (mx - 2.1, yc))
    label(ax, mx + 2.8, yc - 3.2, "$Q_{img} = q$", ha="left", size=S)
    # q path from the fuzzy front end along the top
    ax.plot([29, 29, mx, mx], [yc - 11, 5.5, 5.5, yc - 2.1], color=ACCENT, lw=0.7, ls=DASH, zorder=1)
    ax.add_patch(FancyArrowPatch((mx, yc - 2.6), (mx, yc - 2.1), arrowstyle="-|>,head_length=3,head_width=1.75",
                                 color=ACCENT, linewidth=0.7, mutation_scale=1, shrinkA=0, shrinkB=0, zorder=1))
    ax.text(84, 5.5, "$q$ — image-quality coefficient, Eq. (3), gates the embedding (Eqs. 10–11)",
            ha="center", va="center", fontsize=S, color=ACCENT, zorder=3,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

    # ------------------------------------------------------------------ structured branch, lane 2
    y2 = 46
    box(ax, 2, y2 - 6, 26, 12, "Descriptors $d_n$", ["11 values, image + delineation", "Table 2"],
        title_size=T, line_size=L)
    box(ax, 31, y2 - 6, 36, 12, "Preparation", ["median imputation, z-scoring $\\rightarrow$ $\\tilde d$",
                                              "$Q_{struct}$ = observed fraction — Eq. (10)"], title_size=T, line_size=L)
    arrow(ax, (28, y2), (31, y2))
    node(ax, 73, y2, "×")
    arrow(ax, (67, y2), (70.9, y2))
    label(ax, 73, y2 - 3.4, "$Q_{struct}$", ha="center", size=S)
    # clinical branch (dashed, not in TN3K)
    y2b = 60
    box(ax, 2, y2b - 5.5, 65, 11, "Clinical branch (framework only)",
        ["$\\tilde l_n, \\tilde c_n, \\tilde r_n$ with $Q_{lab}, Q_{clin}, Q_{reg}$ · no data in TN3K"],
        ls=DASH, edge=RULE, title_color=GREY, title_size=T, line_size=L)

    # fusion box
    fx0, fy0, fw, fh = 112, 40, 56, 27
    box(ax, fx0, fy0, fw, fh, "Quality-weighted fusion — Eq. (11)",
        ["$z = [\\,Q_{img}h_{img};\\;Q_{struct}\\tilde d;\\;Q_{img},Q_{struct}\\,]$",
         "128 + 11 + 2 = 141 on TN3K", "general form appends the clinical branch"], fill=TINT, title_size=T, line_size=L)
    arrow(ax, (mx, yc + 2.1), (mx, fy0))                                   # Q_img·h_img ↓
    label(ax, mx + 1.5, (yc + fy0) / 2 + 1, "$Q_{img}h_{img} \\in \\mathbb{R}^{128}$", ha="left", size=S)
    arrow(ax, (75.1, y2), (fx0, y2))                                       # Q_struct·d̃ →
    label(ax, (75 + fx0) / 2, y2 - 2.2, "$Q_{struct}\\tilde d \\in \\mathbb{R}^{11}$", ha="center", size=S)
    dashed_arrow(ax, (67, y2b), (fx0, y2b))
    # z bus to lane 3
    zx, ybus, y3c = fx0 + fw / 2, 72, 90
    ax.plot([zx, zx, 21, 21], [fy0 + fh, ybus, ybus, y3c], color=INK, lw=1.0, zorder=1)
    ax.text(zx + 1.5, (fy0 + fh + ybus) / 2, "$z \\in \\mathbb{R}^{141}$", ha="left", va="center", fontsize=L,
            color=INK, zorder=3)

    # ------------------------------------------------------------------ learners + decision, lane 3
    lx, lw_, lh = 28, 29, 7
    learners = [("DL head $g_\\phi$", "141 $\\rightarrow$ 64 $\\rightarrow$ 1, sigmoid"), ("Random forest", "500 trees, OOB-calibrated"),
                ("XGBoost", "early-stopped on validation"), ("Fuzzy SVM", "RBF, membership $s_n$, Platt")]
    ys = [y3c - 13.5, y3c - 4.5, y3c + 4.5, y3c + 13.5]
    for (t, l_), y in zip(learners, ys):
        box(ax, lx, y - lh / 2, lw_, lh, t, [l_], title_size=5.9, line_size=S, gap=1.2)
        ax.plot([21, 21], [y3c, y], color=INK, lw=1.0, zorder=1)
        arrow(ax, (21, y), (lx, y), color=INK)
    sx_ = lx + lw_ + 10
    node(ax, sx_, y3c, "Σ", r=2.6)
    for y in ys:
        ax.add_patch(FancyArrowPatch((lx + lw_, y), (sx_ - 2.3, y3c), arrowstyle="-|>,head_length=3,head_width=1.75",
                                     color=RULE, linewidth=0.8, mutation_scale=1, shrinkA=0, shrinkB=0, zorder=1,
                                     connectionstyle="arc3,rad=0"))
    label(ax, sx_ + 3, y3c + 11.0, "$p_{ens} = \\Sigma_k\\,\\omega_k p_k$ — Eq. (14);  $\\omega_k \\propto \\exp(-L_k^{val}/\\tau)$, frozen on validation",
          ha="left", size=S)
    label(ax, lx + lw_ / 2, ys[0] - 5.6, "learners fitted on $z$ (training rows), tuned on validation", ha="center",
          size=S, style="italic")
    # SHAP tap on z
    box(ax, 2, y3c + 9, 17, 9, "SHAP on $z$", ["TreeSHAP /", "KernelSHAP"],
        ls=DASH, edge=RULE, title_color=GREY, title_size=5.4, line_size=4.5, gap=0.8)
    ax.plot([19, 21], [y3c + 13.5, y3c + 13.5], color=RULE, lw=0.7, ls=DASH, zorder=1)
    # decision chain
    dx = sx_ + 6
    box(ax, dx, y3c - 7, 30, 14, "Regional prior — Eq. (15)",
        ["$\\mathrm{logit}\\,p_{ens} + \\Delta_{reg}$", "$\\Delta_{reg} = 0$ on TN3K"], title_size=T, line_size=L)
    arrow(ax, (sx_ + 2.6, y3c), (dx, y3c))
    node(ax, dx + 36, y3c, "σ")
    arrow(ax, (dx + 30, y3c), (dx + 33.9, y3c))
    label(ax, dx + 36, y3c - 3.4, "$p_{final}$", ha="center", size=S)
    box(ax, dx + 42, y3c - 7, 28, 14, "Decision — Eq. (16)", ["$\\hat y = 1[\\,p_{final} \\geq t^{*}]$",
                                                            "$t^{*}$: sensitivity ≥ 0.95 on validation"],
        fill=TINT, title_size=T, line_size=L)
    arrow(ax, (dx + 38.1, y3c), (dx + 42, y3c))

    # ------------------------------------------------------------------ (b) fuzzy front end
    yb = 116
    panel_title(ax, 2, yb - 4.5, "(b)", "Fuzzy front end, Eqs. (1)–(8); every map is H × W")
    chain = [("min–max", "$\\tilde x$ · Eq. (1)"), ("local stats", "$c, \\sigma$ on $W_p$ · (2)"),
             ("descriptors", "$T, K, q$ · (3)"), ("membership", "$\\mu(p)$ · (4)"),
             ("radius, weight", "$r(p), w(p)$ · (5)–(6)"), ("weighted ε / δ", "$\\varepsilon_w, \\delta_w$ · (7)"),
             ("open / close", "$\\lambda O + (1-\\lambda) C$ · (8)")]
    bw, bh, gap = 14.8, 10, 2.1
    x = 2
    for i, (t, l_) in enumerate(chain):
        acc = i >= 2
        box(ax, x, yb, bw, bh, t, [l_], fill=ACCENT_FILL if acc else "white", edge=ACCENT if acc else RULE,
            title_color=ACCENT if acc else INK, title_size=5.6, line_size=4.7, gap=1.0)
        if i:
            arrow(ax, (x - gap, yb + bh / 2), (x, yb + bh / 2))
        x += bw + gap
    label(ax, 2 + 2 * (bw + gap) + bw / 2, yb - 1.6, "$T$ texture, $K$ contrast, $q$ quality", ha="center", size=4.4)
    label(ax, 2, yb + bh + 3.2, "$\\tilde x$ and $F$ become the two input channels of (a); $q$ becomes $Q_{img}$; "
                                "$\\alpha, \\beta, \\gamma, \\lambda$ selected on validation by Dice.", ha="left", size=S)

    # ------------------------------------------------------------------ (c) residual block
    cx0 = 132
    panel_title(ax, cx0, yb - 4.5, "(c)", "Basic block of (a)")
    bx, by = cx0 + 2, yb
    steps = [("3×3 conv, BN, ReLU", CONV_FILL, CONV_EDGE), ("3×3 conv, BN", CONV_FILL, CONV_EDGE)]
    cw, ch = 24, 4.6
    arrow(ax, (bx + cw / 2, by - 1.5), (bx + cw / 2, by))
    for i, (t, f, e) in enumerate(steps):
        yy = by + i * (ch + 2.2)
        ax.add_patch(FancyBboxPatch((bx, yy), cw, ch, boxstyle="round,pad=0,rounding_size=0.7", facecolor=f,
                                    edgecolor=e, linewidth=0.8, zorder=2))
        ax.text(bx + cw / 2, yy + ch / 2, t, ha="center", va="center", fontsize=S, color=INK, zorder=3)
        if i:
            arrow(ax, (bx + cw / 2, yy - 2.2), (bx + cw / 2, yy))
    yplus = by + 2 * (ch + 2.2) + 1.2
    node(ax, bx + cw / 2, yplus, "+", r=1.7, size=6.5)
    arrow(ax, (bx + cw / 2, by + 2 * ch + 2.2), (bx + cw / 2, yplus - 1.7))
    ax.plot([bx + cw / 2, bx + cw + 4, bx + cw + 4], [by - 1.5, by - 1.5, yplus], color=RULE, lw=0.8, zorder=1)
    arrow(ax, (bx + cw + 4, yplus), (bx + cw / 2 + 1.7, yplus))
    label(ax, bx + cw + 5, (by - 1.5 + yplus) / 2, "identity\n(1×1 conv, /2\nat stage entry)", ha="left", size=4.6)
    label(ax, bx + cw / 2, yplus + 3.6, "ReLU", ha="center", size=S, color=INK)

    # ------------------------------------------------------------------ legend
    yl = 139.5
    def key(x, text, fill="white", edge=RULE, ls="-", lw=0.8):
        ax.add_patch(FancyBboxPatch((x, yl - 1.5), 4.5, 3, boxstyle="round,pad=0,rounding_size=0.5", zorder=2,
                                    linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls))
        label(ax, x + 5.8, yl, text, ha="left", size=5.4)
    key(2, "fuzzy processing (this work)", ACCENT_FILL, ACCENT, lw=1.0)
    key(42, "convolution", CONV_FILL, CONV_EDGE)
    key(64, "dense / pooling", FC_FILL, FC_EDGE)
    key(90, "learners, fusion, decision")
    key(128, "defined by the framework, no data in TN3K", ls=DASH)
    print(save(fig, "fig2_architecture", a.out))


if __name__ == "__main__":
    main()
