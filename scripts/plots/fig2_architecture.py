#!/usr/bin/env python
"""Figure 2 — the eight layers of TAMES AI (Section 4) with the equation numbers of Section 3; matplotlib patches only.
    python scripts/plots/fig2_architecture.py"""
import sys; sys.path.insert(0, "scripts/plots")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _style import setup, save, W2, MM, INK, INK2, TINT

setup()
H = 128; L = 32; LX = L - 5  # left margin; label right edge 5 mm (≈ 14 pt) before the boxes
fig, ax = plt.subplots(figsize=(W2, H * MM)); ax.set_axis_off(); ax.grid(False); ax.set_xlim(0, 170); ax.set_ylim(2, H)
T, S = 6.2, 5.3
W1_, W2_, W3_ = 40, 44, 44; X2 = L + W1_ + 4; X3 = X2 + W2_ + 4; XR = X3 + W3_  # three columns, right edge XR

def box(x, y, w, h, title, sub=None, fc="white", ec=INK2, lw=0.6, ts=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec, lw=lw))
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=ts or T, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=S, color=INK2)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=T, fontweight="bold", color=INK)

def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, color=INK2, lw=0.6, shrinkA=0, shrinkB=0))

def label(yc, text, color=INK2, bold=False):
    ax.text(LX, yc, text, ha="right", va="center", fontsize=5.8, color=color, fontweight="bold" if bold else "normal", linespacing=1.25)

rows = {}
y = H - 11; rows[1] = y
box(L, y, W1_, 9, "Ultrasound image $I_n$", "B-mode frame, nodule-centred crop"); box(X2, y, W2_, 9, "Descriptors $d_n$", "from image + delineation (TN3K)")
box(X3, y, W3_, 9, "Lab. $l_n$, clinical $c_n$, regional $r_n$", "defined for deployment; not in TN3K"); label(y + 4.5, "Data collection")
y = H - 27; rows[2] = y
ax.add_patch(FancyBboxPatch((L - 1.5, y - 4.5), W1_ + 3, 15.5, boxstyle="round,pad=0,rounding_size=1.4", fc=TINT, ec="#0072B2", lw=0.9))
box(L + 1, y + 5, 18, 5, "ATCFM  $\\mu(p)$", fc="white"); box(L + 21, y + 5, 18, 5, "AFMO  $F$", fc="white")
ax.text(L + W1_ / 2, y + 1.6, "Eqs. (1)–(4): $\\tilde{x}, c, \\sigma, T, K, q, \\mu$", ha="center", va="center", fontsize=S, color=INK2)
ax.text(L + W1_ / 2, y - 1.6, "Eqs. (5)–(8): $r(p), w(p), \\varepsilon_w, \\delta_w, O, C, F$", ha="center", va="center", fontsize=S, color=INK2)
label(y + 3.25, "Adaptive fuzzy layer\n(this work)", "#0072B2", True)
box(X2, y + 0.5, W2_, 9, "Preparation + quality", "impute, z-score; $Q_{img}$, $Q_{struct}$ (Eq. 10)")
box(X3, y + 0.5, W3_, 9, "Preparation + quality", "$Q_{lab}$, $Q_{clin}$, $Q_{reg}$ (Eq. 10)")
y = H - 44; rows[3] = y
box(L, y, W1_ + 4, 9, "Backbone $f_\\theta$: $[\\tilde{x}, F] \\rightarrow h_{img} \\in \\mathbb{R}^{128}$", "ResNet-18, two input channels (Eq. 9)", ts=T - 0.5)
box(L + W1_ + 8, y, XR - (L + W1_ + 8), 9, "Fusion  $z = [Q_{img} h_{img} ; Q_{struct} \\tilde{d} ; Q_{img}, Q_{struct}] \\in \\mathbb{R}^{141}$", "general form with $\\tilde{l}, \\tilde{c}, \\tilde{r}$ (Eq. 11)", ts=T - 0.3)
label(y + 4.5, "Data preparation\nand fusion")
y = H - 60; rows[4] = y
cw = (XR - L - 3 * 4) / 4
for i, (t, s) in enumerate([("DL head $g_\\phi$", "$141 \\rightarrow 64 \\rightarrow 1$"), ("Random forest", "OOB-calibrated"), ("XGBoost", "logistic objective"), ("Fuzzy SVM", "$s_n$ (Eqs. 12–13)")]):
    box(L + i * (cw + 4), y, cw, 9, t, s)
label(y + 4.5, "Hybrid learners")
y = H - 75; rows[5] = y
box(L, y, XR - L, 9, "Adaptive ensemble   $p_{ens} = \\Sigma_k\\, \\omega_k p_k$,   $\\omega_k = \\mathrm{softmax}(-L_k^{val} / \\tau)$", "Eq. (14); $\\tau$ and $\\omega$ fixed on validation"); label(y + 4.5, "Adaptive ensemble")
y = H - 90; rows[6] = y
hw = (XR - L - 6) / 2
box(L, y, hw, 9, "Regional prior   $\\Delta_{reg} = \\mathrm{logit}\\,\\pi_{reg} - \\mathrm{logit}\\,\\pi_{train}$", "Eq. (15); $\\Delta_{reg} = 0$ on TN3K")
box(L + hw + 6, y, hw, 9, "Decision   $p_{final} = \\sigma(\\mathrm{logit}\\,p_{ens} + \\Delta_{reg})$,  $\\hat{y} = 1[p_{final} \\geq t^*]$", "Eq. (16); $t^*$: sensitivity $\\geq$ 0.95 on validation", ts=T - 1.0); label(y + 4.5, "Regional prior\nand decision")
y = H - 105; rows[7] = y
box(L, y, hw, 9, "SHAP   $\\phi_j^{ens} = \\Sigma_k\\, \\omega_k \\phi_j^k$", "TreeSHAP / KernelSHAP on $z$; rank stability")
box(L + hw + 6, y, hw, 9, "Grad-CAM on the last residual block", "deletion / insertion faithfulness vs random saliency"); label(y + 4.5, "Explainable AI")
y = H - 120; rows[8] = y
box(L, y, XR - L, 9, "Output: label $\\hat{y}$, probability $p_{final}$, feature attribution, saliency map"); label(y + 4.5, "Decision support")
c1, c2, c3 = L + W1_ / 2, X2 + W2_ / 2, X3 + W3_ / 2
arrow(c1, rows[1], c1, rows[2] + 11); arrow(c2, rows[1], c2, rows[2] + 9.5); arrow(c3, rows[1], c3, rows[2] + 9.5)
arrow(c1, rows[2] - 4.5, c1, rows[3] + 9); arrow(c2, rows[2] + 0.5, c2, rows[3] + 9); arrow(c3, rows[2] + 0.5, c3, rows[3] + 9)
arrow(L + W1_ + 4, rows[3] + 4.5, L + W1_ + 8, rows[3] + 4.5)
for i in range(4):
    xc = L + i * (cw + 4) + cw / 2; arrow(xc, rows[3], xc, rows[4] + 9); arrow(xc, rows[4], xc, rows[5] + 9)
xl, xr = L + hw / 2, L + hw + 6 + hw / 2
arrow(xl, rows[5], xl, rows[6] + 9); arrow(xr, rows[5], xr, rows[6] + 9); arrow(L + hw, rows[6] + 4.5, L + hw + 6, rows[6] + 4.5)
arrow(xl, rows[6], xl, rows[7] + 9); arrow(xr, rows[6], xr, rows[7] + 9); arrow(xl, rows[7], xl, rows[8] + 9); arrow(xr, rows[7], xr, rows[8] + 9)
save(fig, "fig2_architecture")
