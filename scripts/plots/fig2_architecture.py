#!/usr/bin/env python
"""Figure 2 — the eight layers of TAMES AI (Section 4) with the equation numbers of Section 3; matplotlib patches only.
    python scripts/plots/fig2_architecture.py"""
import sys; sys.path.insert(0, "scripts/plots")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _style import setup, save, W2, MM, INK, INK2, TINT

setup()
H = 128; L = 24  # left margin for layer labels
fig, ax = plt.subplots(figsize=(W2, H * MM)); ax.set_axis_off(); ax.grid(False); ax.set_xlim(0, 170); ax.set_ylim(0, H)
T, S = 6.2, 5.3

def box(x, y, w, h, title, sub=None, fc="white", ec=INK2, lw=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec, lw=lw))
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=T, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=S, color=INK2)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=T, fontweight="bold", color=INK)

def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, color=INK2, lw=0.6, shrinkA=0, shrinkB=0))

def label(y, text, color=INK2, bold=False):
    ax.text(L - 2, y, text, ha="right", va="center", fontsize=5.8, color=color, fontweight="bold" if bold else "normal")

R = r"$\mathbb{R}$"
rows = {}
y = H - 11; rows[1] = y
box(L, y, 44, 9, "Ultrasound image $I_n$", "B-mode frame, nodule-centred crop"); box(L + 47, y, 44, 9, "Descriptors $v_n$", "from image + delineation (TN3K)")
box(L + 94, y, 52, 9, "Laboratory $l_n$, clinical $c_n$, regional $r_n$", "defined for deployment; not in TN3K"); label(y + 4.5, "1  Data collection")
y = H - 27; rows[2] = y
ax.add_patch(FancyBboxPatch((L - 1.5, y - 4.5), 47, 15.5, boxstyle="round,pad=0,rounding_size=1.4", fc=TINT, ec="#0072B2", lw=0.9))
box(L + 1, y + 5, 20, 5, "ATCFM  μ(p)", fc="white"); box(L + 23, y + 5, 20, 5, "AFMO  F", fc="white")
ax.text(L + 22, y + 1.6, "Eqs. (1)–(4): x̃, c, σ, T, K, q, μ", ha="center", va="center", fontsize=S, color=INK2)
ax.text(L + 22, y - 1.6, "Eqs. (5)–(8): r(p), w(p), ε_w, δ_w, O, C, F", ha="center", va="center", fontsize=S - 0.3, color=INK2)
label(y + 5, "2  Adaptive fuzzy layer", "#0072B2", True); label(y + 1.3, "(this work)", "#0072B2")
box(L + 47, y + 0.5, 44, 9, "Preparation + quality", "imputation, z-scoring, $Q_{img}$, $Q_{struct}$ — Eq. (10)")
box(L + 94, y + 0.5, 52, 9, "Preparation + quality", "$Q_{lab}$, $Q_{clin}$, $Q_{reg}$ — Eq. (10)"); label(y - 6, "3  Data preparation")
y = H - 44; rows[3] = y
box(L, y, 48, 9, "Backbone $f_\\theta$: $[\\tilde{x}, F]$ " + r"$\rightarrow h_{img} \in \mathbb{R}^{128}$", "ResNet-18, two input channels — Eq. (9)")
box(L + 51, y, 95, 9, "Fusion  z = [$Q_{img} h_{img}$ ; $Q_{struct} \\tilde{v}$ ; $Q_{img}$, $Q_{struct}$] " + r"$\in \mathbb{R}^{141}$", "general form with $\\tilde{l}$, $\\tilde{c}$, $\\tilde{r}$ — Eq. (11)")
label(y + 4.5, "4  Fusion")
y = H - 60; rows[4] = y
for i, (t, s) in enumerate([("DL head $g_\\phi$", "141 $\\rightarrow$ 64 $\\rightarrow$ 1"), ("Random forest", "OOB-calibrated"), ("XGBoost", "logistic objective"), ("Fuzzy SVM", "$s_n$ — Eqs. (12)–(13)")]):
    box(L + i * 37.5, y, 33.5, 9, t, s)
label(y + 4.5, "5  Hybrid learners")
y = H - 75; rows[5] = y
box(L, y, 146, 9, "Adaptive ensemble   $p_{ens} = \\sum_k \\omega_k p_k$,   $\\omega_k$ = softmax(−$L_k^{val}$ / τ)", "Eq. (14); τ and ω fixed on validation"); label(y + 4.5, "6  Ensemble")
y = H - 90; rows[6] = y
box(L, y, 64, 9, "Regional prior   $\\Delta_{reg}$ = logit $\\pi_{reg}$ − logit $\\pi_{train}$", "Eq. (15); $\\Delta_{reg}$ = 0 on TN3K")
box(L + 70, y, 76, 9, "Decision   $p_{final} = \\sigma$(logit $p_{ens}$ + $\\Delta_{reg}$),  ŷ = 1[$p_{final} \\geq t^*$]", "Eq. (16); t*: sensitivity ≥ 0.95 on validation"); label(y + 4.5, "7  Prior + decision")
y = H - 105; rows[7] = y
box(L, y, 70, 9, "SHAP   $\\phi_j^{ens} = \\sum_k \\omega_k \\phi_j^k$", "TreeSHAP / KernelSHAP on z; rank stability")
box(L + 76, y, 70, 9, "Grad-CAM on the last residual block", "deletion / insertion faithfulness vs random saliency"); label(y + 4.5, "8  Explainable AI")
y = H - 120; rows[8] = y
box(L, y, 146, 9, "Output: label ŷ, probability $p_{final}$, feature attribution, saliency map"); label(y + 4.5, "9  Decision support")
# arrows (column centres: 46, 93, 144)
c1, c2, c3 = L + 22, L + 69, L + 120
arrow(c1, rows[1], c1, rows[2] + 11); arrow(c2, rows[1], c2, rows[2] + 9.5); arrow(c3, rows[1], c3, rows[2] + 9.5)
arrow(c1, rows[2] - 4.5, c1, rows[3] + 9); arrow(c2, rows[2] + 0.5, c2, rows[3] + 9); arrow(c3, rows[2] + 0.5, c3, rows[3] + 9)
arrow(L + 48, rows[3] + 4.5, L + 51, rows[3] + 4.5)
for i in range(4):
    xc = L + i * 37.5 + 16.75; arrow(xc, rows[3], xc, rows[4] + 9); arrow(xc, rows[4], xc, rows[5] + 9)
arrow(L + 32, rows[5], L + 32, rows[6] + 9); arrow(L + 108, rows[5], L + 108, rows[6] + 9); arrow(L + 64, rows[6] + 4.5, L + 70, rows[6] + 4.5)
arrow(L + 32, rows[6], L + 32, rows[7] + 9); arrow(L + 108, rows[6], L + 108, rows[7] + 9)
arrow(L + 35, rows[7], L + 35, rows[8] + 9); arrow(L + 111, rows[7], L + 111, rows[8] + 9)
save(fig, "fig2_architecture")
