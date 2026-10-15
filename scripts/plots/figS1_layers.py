#!/usr/bin/env python
"""Supplementary Figure S1 — layered view (architecture of TAMES AI as eight layers; 170 mm, double column).

Solid boxes are evaluated on TN3K; dashed boxes are defined by the framework but not
available in TN3K; the blue box is the contribution of this work. Equation numbers refer to
Section 3 of the manuscript.
    python scripts/plots/fig2_architecture.py [--out paper/figures]
"""
import argparse
import sys
from pathlib import Path

from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle_diagram import (ACCENT, ACCENT_FILL, GREY, INK, RULE, TINT, arrow, box, canvas,  # noqa: E402
                              label, save, setup_font)

ROOT = Path(__file__).resolve().parents[2]
T, L = 7.0, 5.9  # title / line sizes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    a = ap.parse_args()
    setup_font()
    W, H = 170, 150
    fig, ax = canvas(W, H)
    LX = 29.5                                    # right edge of the label column
    A = (33, 45); B = (81, 41); C = (125, 43)    # (x, width) of the three input columns
    cA, cB, cC = A[0] + A[1] / 2, B[0] + B[1] / 2, C[0] + C[1] / 2
    right = C[0] + C[1]
    dashed = dict(ls=(0, (2.2, 1.6)), edge=RULE, title_color=GREY)

    # ---- R1 data collection
    y1, h1 = 2, 11.5
    box(ax, A[0], y1, A[1], h1, "Ultrasound image $I_n$", ["B-mode frame, 224 × 224 nodule-centred crop"],
        title_size=T, line_size=L)
    box(ax, B[0], y1, B[1], h1, "Sonographic descriptors $d_n$", ["11 values from image + delineation",
                                                           "computable TI-RADS counterparts"], title_size=T, line_size=L)
    box(ax, C[0], y1, C[1], h1, "Clinical branch", ["laboratory $l_n$, clinical $c_n$, regional $r_n$",
                                             "defined for deployment · not in TN3K"], title_size=T, line_size=L, **dashed)
    label(ax, LX, y1 + h1 / 2, "Data collection")

    # ---- R2 adaptive fuzzy layer | preparation
    y2, h2 = 20, 15
    box(ax, A[0], y2, A[1], h2, "Adaptive fuzzy layer (this work)",
        ["ATCFM membership $\\mu(p)$ — Eqs. (1)–(4)", "AFMO map $F=\\lambda O+(1-\\lambda)C$ — Eqs. (5)–(8)",
         "image-quality coefficient $q$ — Eq. (3)"], fill=ACCENT_FILL, edge=ACCENT, lw=1.1,
        title_size=T, line_size=L, title_color=ACCENT)
    box(ax, B[0], y2, B[1], h2, "Preparation and quality", ["median imputation, z-scoring",
                                                     "$Q_{struct}$ = observed fraction — Eq. (10)"], title_size=T, line_size=L)
    box(ax, C[0], y2, C[1], h2, "Preparation and quality", ["$Q_{lab},\\,Q_{clin},\\,Q_{reg}$ — Eq. (10)",
                                                     "not evaluated in this study"], title_size=T, line_size=L, **dashed)
    label(ax, LX, y2 + h2 / 2, "Adaptive fuzzy layer\nData preparation")

    # ---- R3 backbone | fusion
    y3, h3 = 41, 12.5
    box(ax, A[0], y3, A[1], h3, "Backbone $f_\\theta$ (ResNet-18)", ["two input channels $[\\tilde x, F]$",
                                                              "128-d embedding $h_{img}$ — Eq. (9)"], title_size=T, line_size=L)
    box(ax, B[0], y3, right - B[0], h3, "Quality-weighted fusion — Eq. (11)",
        ["$z=[\\,Q_{img}h_{img};\\;Q_{struct}\\tilde d;\\;Q_{img},Q_{struct}\\,]$, 141-d on TN3K",
         "general form appends $\\tilde l,\\tilde c,\\tilde r$ with $Q_{lab},Q_{clin},Q_{reg}$"], title_size=T, line_size=L)
    label(ax, LX, y3 + h3 / 2, "Fusion")

    # ---- R4 learners
    y4, h4 = 59.5, 11.5
    lw4 = (right - A[0] - 3 * 3) / 4
    lx = [A[0] + i * (lw4 + 3) for i in range(4)]
    lc = [x + lw4 / 2 for x in lx]
    box(ax, lx[0], y4, lw4, h4, "DL head $g_\\phi$", ["two-layer head on $z$", "141 $\\rightarrow$ 64 $\\rightarrow$ 1, sigmoid"], title_size=T, line_size=L)
    box(ax, lx[1], y4, lw4, h4, "Random forest", ["500 trees", "out-of-bag calibration"], title_size=T, line_size=L)
    box(ax, lx[2], y4, lw4, h4, "XGBoost", ["gradient-boosted trees", "logistic objective"], title_size=T, line_size=L)
    box(ax, lx[3], y4, lw4, h4, "Fuzzy SVM", ["sample membership $s_n$", "Eqs. (12)–(13)"], title_size=T, line_size=L)
    label(ax, LX, y4 + h4 / 2, "Hybrid learners")

    # ---- R5 ensemble
    y5, h5 = 77, 10.5
    box(ax, A[0], y5, right - A[0], h5, "Adaptive ensemble — Eq. (14)",
        ["$p_{ens}=\\mathrm{\\Sigma}_k\\,\\omega_k p_k$,   $\\omega_k\\propto\\exp(-L_k^{val}/\\tau)$   ·   $\\tau$ and $\\omega_k$ fixed on validation"],
        title_size=T, line_size=L)

    # ---- R6 prior | decision
    y6, h6 = 93.5, 12.5
    px, pw = A[0], 60
    dx = px + pw + 3; dw = right - dx
    box(ax, px, y6, pw, h6, "Regional prior — Eq. (15)",
        ["$\\Delta_{reg}=\\mathrm{logit}\\,\\pi_{reg}-\\mathrm{logit}\\,\\pi_{train}$", "$\\Delta_{reg}=0$ on TN3K"],
        title_size=T, line_size=L)
    box(ax, dx, y6, dw, h6, "Decision rule — Eq. (16)",
        ["$p_{final}=\\sigma(\\mathrm{logit}\\,p_{ens}+\\Delta_{reg})$,   $\\hat y=1[\\,p_{final}\\geq t^*]$",
         "$t^*$: sensitivity ≥ 0.95 on validation"], title_size=T, line_size=L)
    label(ax, LX, (y5 + y6 + h6) / 2, "Ensemble and decision")

    # ---- R7 explainable AI
    y7, h7 = 112, 12.5
    box(ax, px, y7, pw, h7, "SHAP attribution", ["TreeSHAP / KernelSHAP on $z$;  $\\phi^{ens}_j=\\mathrm{\\Sigma}_k\\,\\omega_k\\phi^k_j$",
                                                 "top-10 rank stability over resamples"], title_size=T, line_size=L)
    box(ax, dx, y7, dw, h7, "Grad-CAM", ["last residual block of $f_\\theta$",
                                         "deletion / insertion faithfulness vs random saliency"], title_size=T, line_size=L)
    label(ax, LX, y7 + h7 / 2, "Explainable AI")

    # ---- R8 output
    y8, h8 = 130.5, 9.5
    box(ax, A[0], y8, right - A[0], h8, "Decision-support output",
        ["label $\\hat y$  ·  probability $p_{final}$  ·  feature attribution  ·  saliency map"],
        fill=TINT, title_size=T, line_size=L)
    label(ax, LX, y8 + h8 / 2, "Decision support")

    # ---- connectors
    dash = dict(color=RULE)
    arrow(ax, (cA, y1 + h1), (cA, y2)); arrow(ax, (cB, y1 + h1), (cB, y2))
    ax.plot([cC, cC], [y1 + h1, y2 - 1.2], color=RULE, lw=0.8, ls=(0, (2.2, 1.6)))
    arrow(ax, (cC, y2 - 1.2), (cC, y2))
    arrow(ax, (cA, y2 + h2), (cA, y3)); arrow(ax, (cB, y2 + h2), (cB, y3))
    ax.plot([cC, cC], [y2 + h2, y3 - 1.2], color=RULE, lw=0.8, ls=(0, (2.2, 1.6)))
    arrow(ax, (cC, y3 - 1.2), (cC, y3))
    arrow(ax, (A[0] + A[1], y3 + h3 / 2), (B[0], y3 + h3 / 2))                 # backbone → fusion
    fx = B[0] + (right - B[0]) / 2
    ybus = y3 + h3 + 3
    ax.plot([fx, fx], [y3 + h3, ybus], color=RULE, lw=0.8)
    ax.plot([lc[0], lc[3]], [ybus, ybus], color=RULE, lw=0.8)
    for x in lc:
        arrow(ax, (x, ybus), (x, y4))
    for x in lc:
        arrow(ax, (x, y4 + h4), (x, y5))
    dcx = dx + dw / 2
    arrow(ax, (dcx, y5 + h5), (dcx, y6))                                        # ensemble → decision
    arrow(ax, (px + pw, y6 + h6 / 2), (dx, y6 + h6 / 2))                        # prior → decision
    ybus2 = y6 + h6 + 2.8
    scx = px + pw / 2
    ax.plot([dcx, dcx], [y6 + h6, ybus2], color=RULE, lw=0.8)
    ax.plot([scx, dcx], [ybus2, ybus2], color=RULE, lw=0.8)
    arrow(ax, (scx, ybus2), (scx, y7)); arrow(ax, (dcx, ybus2), (dcx, y7))
    ybus3 = y7 + h7 + 3
    ax.plot([scx, scx], [y7 + h7, ybus3], color=RULE, lw=0.8)
    ax.plot([dcx, dcx], [y7 + h7, ybus3], color=RULE, lw=0.8)
    ax.plot([scx, dcx], [ybus3, ybus3], color=RULE, lw=0.8)
    ox = A[0] + (right - A[0]) / 2
    arrow(ax, (ox, ybus3), (ox, y8))

    # ---- legend
    yl = 144.5
    def key(x, text, **kw):
        ax.add_patch(FancyBboxPatch((x, yl - 1.6), 5, 3.2, boxstyle="round,pad=0,rounding_size=0.6", zorder=2,
                                    linewidth=kw.get("lw", 0.8), edgecolor=kw.get("edge", RULE),
                                    facecolor=kw.get("fill", "white"), linestyle=kw.get("ls", "-")))
        label(ax, x + 6.5, yl, text, ha="left", size=6.0)
    key(A[0], "evaluated on TN3K")
    key(A[0] + 38, "defined by the framework, not available in TN3K", ls=(0, (2.2, 1.6)))
    key(A[0] + 104, "contribution of this work", fill=ACCENT_FILL, edge=ACCENT, lw=1.0)
    print(save(fig, "figS1_layers", a.out))


if __name__ == "__main__":
    main()
