#!/usr/bin/env python
"""Figure 2 — computational architecture of TAMES AI (170 mm, double column).

(a) The model as a data-flow graph with per-image tensor shapes on the edges:
    native B-mode frame I_n and delineation M_n → fuzzy front end on the native H × W frame
    → nodule crop (1.5 × bbox of M_n) resized to 224² → f_θ = ResNet-18 trunk + fc embedding
    (Eq. 9) → quality gate Q_img = q; structured branch d_n → d̃, Q_struct (Eq. 10);
    quality-weighted fusion z ∈ R^141 (Eq. 11) → four learners (DL head trained end-to-end
    with f_θ; RF, XGB, FSVM fitted on z) → adaptive ensemble (Eq. 14) → prior shift
    (Eqs. 15–16) → mean over five seeds → threshold t* → output.
(b) The fuzzy front end, Eqs. (1)–(8), as a dependency graph, with the real maps of one TN3K
    test image computed here by the repository code (tames.fuzzy) with the parameters the
    manuscript reports as selected (α = 2, β = 0.5, γ = 2, λ = 0, bright-side polarity,
    SNR_0 = median training SNR). The image is te_0450, the test image whose Otsu(F)-vs-M_n
    Dice is closest to the test-set median (rule applied once over all 614 test images).
    If the raw TN3K files are not available, or with --no-maps, (b) is drawn without maps.

Stroke grammar (one meaning per style): solid grey arrow = data flow evaluated on TN3K;
thin solid blue = scalar quality coefficient (q, Q_struct); dashed grey = defined by the
framework, no data in TN3K; dotted grey = evaluation-time tap (SHAP, Grad-CAM).
All text ≥ 5.5 pt; nothing is drawn beyond x = 168 mm; the saved figure is 170 mm wide.

    python scripts/plots/fig2_architecture.py [--out paper/figures] [--no-maps]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle_diagram import ACCENT, ACCENT_FILL, GREY, INK, RULE, TINT, canvas, save, setup_font  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAD = 0.02 * 25.4                                 # pad_inches of figstyle_diagram.save, mm
W, H = 170.0 - 2 * PAD, 166.0 - 2 * PAD            # axes; saved page = W + 2·PAD = 170 mm wide
XMAX = 168.0                                      # right limit for every drawn element
CONV_FILL, CONV_EDGE = "#fdf1e7", "#D55E00"       # convolution (Okabe–Ito vermilion)
FC_FILL, FC_EDGE = "#e8f4ee", "#009E73"           # dense / pooling (Okabe–Ito bluish green)
PANEL = "#f8f8f6"
MASK_C = "#E69F00"                                # delineation contour (Okabe–Ito orange)
T, L, S = 7.0, 6.0, 5.5                           # title / line / shape-label sizes (pt)
LW = 0.8
DASH = (0, (3.0, 1.8))
DOT = (0, (0.8, 1.2))
MM_PER_PT = 25.4 / 72
STYLE = {"data": dict(color=RULE, ls="-"), "q": dict(color=ACCENT, ls="-"),
         "dash": dict(color=RULE, ls=DASH), "tap": dict(color=GREY, ls=DOT)}

# Example image and the parameters the manuscript reports as selected (Section 3.5, 5.1).
EXAMPLE_ID = "0450"
FUZZY_SELECTED = dict(k=7, alpha=2.0, beta=0.5, r0=2, gamma=2.0, r_min=1, r_max=5, lam=0.0)
POLARITY = "high"


# ---------------------------------------------------------------------------- layout registry
class Reg:
    """Remembers every box and text so that the layout can be checked after drawing."""

    def __init__(self):
        self.rects, self.texts = {}, []

    def rect(self, key, x, y, w, h, parent=None):
        assert key not in self.rects, key
        self.rects[key] = (x, y, x + w, y + h, parent)

    def ancestors(self, key):
        out = set()
        while key is not None:
            out.add(key)
            key = self.rects[key][4]
        return out


G = Reg()
AX = None
FIG = None


def rbox(key, x, y, w, h, *, fill="white", edge=RULE, lw=LW, ls="-", radius=1.0, parent=None, z=2):
    AX.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=fill, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=z))
    G.rect(key, x, y, w, h, parent)


_WORD_SUB = re.compile(r"_\{(img|struct|ens|final|reg|lab|clin|val)\}")
_DIGITS = re.compile(r"(?<![\\A-Za-z])(\d+(?:\.\d+)?)")
_PUNCT = re.compile(r"(?<!\\)([\[\];])")


def m(s):
    """Upright digits, brackets and word subscripts inside $…$ (mathtext.default is italic)."""
    parts = s.split("$")
    for i in range(1, len(parts), 2):
        t = _WORD_SUB.sub(r"_{\\mathrm{\1}}", parts[i])
        t = _PUNCT.sub(r"\\mathrm{\1}", t)
        t = _DIGITS.sub(r"\\mathrm{\1}", t)
        parts[i] = t
    return "$".join(parts)


def tx(x, y, s, *, size=S, ha="center", va="center", color=GREY, weight="normal", style="normal",
       parent=None, inside=False, z=4):
    t = AX.text(x, y, m(s), ha=ha, va=va, fontsize=size, color=color, fontweight=weight, fontstyle=style,
                zorder=z)
    G.texts.append((t, parent, inside))
    return t


def box(key, x, y, w, h, title=(), lines=(), *, tsize=T, lsize=L, lcolor=GREY, tcolor=INK, lead=1.22,
        dy=0.0, **kw):
    """Rounded box with bold title line(s) and body lines, stacked and centred vertically."""
    rbox(key, x, y, w, h, **kw)
    title = [title] if isinstance(title, str) else list(title)
    rows = [(s, tsize, "bold", tcolor) for s in title] + [(s, lsize, "normal", lcolor) for s in lines]
    leads = [r[1] * MM_PER_PT * lead for r in rows]
    yy = y + h / 2 - sum(leads) / 2 + dy
    for (s, size, wt, c), ld in zip(rows, leads):
        tx(x + w / 2, yy + ld / 2, s, size=size, weight=wt, color=c, parent=key, inside=True)
        yy += ld


def head(p_from, p_to, color):
    d = np.subtract(p_to, p_from)
    d = d / (np.hypot(*d) + 1e-9)
    AX.add_patch(FancyArrowPatch(tuple(np.subtract(p_to, d * 0.05)), p_to,
                                 arrowstyle="-|>,head_length=3,head_width=1.75", color=color, linewidth=LW,
                                 linestyle="-", mutation_scale=1, shrinkA=0, shrinkB=0, zorder=3))


def path(pts, kind="data", arrow=True, z=1):
    st = STYLE[kind]
    xs, ys = zip(*pts)
    AX.plot(xs, ys, color=st["color"], lw=LW, ls=st["ls"], solid_capstyle="butt", dash_capstyle="butt",
            solid_joinstyle="miter", zorder=z)
    if arrow:
        head(pts[-2], pts[-1], st["color"])


def dot(x, y, color=RULE):
    AX.add_patch(Circle((x, y), 0.45, facecolor=color, edgecolor="none", zorder=3))


def node(key, x, y, sym, r=2.1, size=7.0):
    AX.add_patch(Circle((x, y), r, facecolor="white", edgecolor=INK, linewidth=LW, zorder=3))
    AX.text(x, y - 0.05, sym, ha="center", va="center", fontsize=size, color=INK, zorder=4)
    G.rect(key, x - r, y - r, 2 * r, 2 * r)


def panel_title(x, y, letter, text):
    tx(x, y, letter, size=8, ha="left", color=INK, weight="bold")
    tx(x + 4.6, y, text, size=T, ha="left", color=INK)


def edge_label(x, y, s, **kw):
    return tx(x, y, s, size=S, **kw)


# ---------------------------------------------------------------------------- example maps
def example_maps():
    """Real maps of one TN3K test image, cropped and resized exactly as the pipeline does."""
    try:
        import cv2
        sys.dont_write_bytecode = True                              # leave the repository untouched
        sys.path.insert(0, str(ROOT))
        from tames.data import load_image, load_mask, nodule_crop
        from tames.fuzzy import dice, fuzzy_maps, otsu_binarize
        raw = ROOT / "data" / "raw" / "tn3k"
        img = load_image(raw / "test-image" / f"{EXAMPLE_ID}.jpg")
        gt = load_mask(raw / "test-mask" / f"{EXAMPLE_ID}.jpg")
        snr0 = json.load(open(ROOT / "data" / "processed" / "snr0.json"))["snr0"]
        m = fuzzy_maps(img, FUZZY_SELECTED, snr0)                  # Eqs. (1)–(8), native H × W
        seg = otsu_binarize(m["F"], POLARITY)
        y0, y1, x0, x1 = nodule_crop(gt, 1.5)                       # main experiment: crop from M_n

        def rs(a, nearest=False):
            return cv2.resize(np.asarray(a, np.float32)[y0:y1, x0:x1], (224, 224),
                              interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
        return {"x": rs(m["x"]), "sigma": rs(m["sigma"]), "mu": rs(m["mu"]), "r": rs(m["r"], True),
                "F": rs(m["F"]), "seg": rs(seg, True), "gt": rs(gt, True), "q": float(m["q"]),
                "dice": float(dice(seg, gt)), "shape": img.shape}
    except Exception as e:                                          # no raw data → no thumbnails
        print(f"[fig2] example maps unavailable ({type(e).__name__}: {e}); panel (b) without maps")
        return None


def thumb(key, x, y, s, img, cmap, contour=None, vmin=None, vmax=None):
    a = FIG.add_axes([x / W, 1 - (y + s) / H, s / W, s / H])
    vmin = float(img.min()) if vmin is None else vmin
    vmax = float(img.max()) if vmax is None else vmax
    a.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="antialiased")
    if contour is not None:
        a.contour(contour, levels=[0.5], colors=[MASK_C], linewidths=0.7)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_linewidth(0.5); sp.set_color(RULE)
    G.rect(key, x, y, s, s)
    return vmin, vmax


# ---------------------------------------------------------------------------- panel (a)
def panel_a():
    panel_title(2, 2.2, "(a)", "TAMES AI as a data-flow graph; tensor shapes per image on the edges")
    yc = 24.0

    # inputs and fuzzy front end -------------------------------------------------------
    box("input", 2, 15, 14.2, 18, "Input", ["$I_n$: B-mode,", "H × W, 8-bit", "$M_n$: nodule", "delineation"],
        fill=TINT)
    box("fuzzy", 18.5, 12, 20, 24, ["Fuzzy", "front end"], ["ATCFM: $\\mu$", "AFMO: $F$", "quality: $q$",
                                                            "Eqs. (1)–(8)", "details in (b)"],
        fill=ACCENT_FILL, edge=ACCENT, lw=1.0)
    path([(16.2, yc), (18.5, yc)])
    box("crop", 46.5, 16, 12, 16, "Crop", ["1.5× bbox", "of $M_n$,", "resize", "to 224²"])
    path([(38.5, yc), (46.5, yc)])
    edge_label(42.5, yc - 4.3, "$\\tilde x, F$")
    edge_label(42.5, yc - 1.9, "H × W")

    # M_n → crop and (x̃, M_n) → descriptors
    path([(12.0, 33.0), (12.0, 39.3), (52.5, 39.3), (52.5, 32.0)])
    edge_label(33.0, 41.0, "$M_n$ (E1: Otsu$(F)$)")
    path([(5.5, 33.0), (5.5, 48.0)])
    edge_label(6.7, 44.6, "$I_n$, $M_n$ (H × W)", ha="left")

    # f_θ container ------------------------------------------------------------------
    cx0, cy0, ch = 66.0, 7.0, 33.0
    sw, stw, gap, pw, fw = 10.4, 9.6, 2.5, 6.0, 12.4
    cw = 1.5 + sw + 4 * (gap + stw) + gap + pw + gap + fw + 1.5
    rbox("ftheta", cx0, cy0, cw, ch, fill=PANEL, edge=RULE, lw=0.7, radius=1.4, z=0.5)
    tx(cx0 + 1.8, 9.7, "$f_\\theta$ — Eq. (9): ResNet-18 trunk + fc embedding", size=T, ha="left",
       color=INK, weight="bold", parent="ftheta", inside=True)
    ylab = 13.0                                                     # one baseline for all shapes
    path([(58.5, yc), (cx0 + 1.5, yc)])
    edge_label(62.25, yc - 4.3, "[$\\tilde x$, $F$]")
    edge_label(62.25, yc - 1.9, "224²×2")

    x = cx0 + 1.5
    box("stem", x, yc - 9, sw, 18, (), ["7×7 conv", "64, /2", "BN, ReLU", "3×3 max", "pool, /2"],
        lsize=S, lcolor=INK, fill=CONV_FILL, edge=CONV_EDGE, radius=0.8, parent="ftheta")
    x += sw
    shapes = ["56²×64", "56²×64", "28²×128", "14²×256", "7²×512"]
    heights = [18, 15, 12, 10]
    tap_x = None
    for i, h in enumerate(heights):
        ex = x + gap / 2
        path([(x, yc), (x + gap, yc)])
        edge_label(ex, ylab, shapes[i], parent="ftheta")
        x += gap
        lines = ["basic", "block ×2"] + (["/2"] if i else [])
        box(f"stage{i + 1}", x, yc - h / 2, stw, h, (), lines, lsize=S, lcolor=INK, fill=CONV_FILL,
            edge=CONV_EDGE, radius=0.8, parent="ftheta")
        x += stw
    # stage 4 → avg pool, Grad-CAM tap on the layer4 output
    tap_x = x + gap / 2
    path([(x, yc), (x + gap, yc)])
    edge_label(tap_x, ylab, shapes[4], parent="ftheta")
    x += gap
    box("pool", x, yc - 4, pw, 8, (), ["avg", "pool"], lsize=S, lcolor=INK, fill=FC_FILL, edge=FC_EDGE,
        radius=0.8, parent="ftheta")
    x += pw
    path([(x, yc), (x + gap, yc)])
    edge_label(x + gap / 2, ylab, "512", parent="ftheta")
    x += gap
    box("fc", x, yc - 7.5, fw, 15, (), ["fc, 128", "ReLU,", "dropout 0.3"], lsize=S, lcolor=INK,
        fill=FC_FILL, edge=FC_EDGE, radius=0.8, parent="ftheta")
    fc_r = x + fw
    path([(tap_x, yc), (tap_x, 34.6)], kind="tap", arrow=False)
    box("gradcam", tap_x - 6.2, 34.6, 12.4, 3.9, (), ["Grad-CAM"], lsize=S, lcolor=GREY, ls=DOT, edge=GREY,
        radius=0.6, parent="ftheta")

    # quality gate × Q_img and the q path -------------------------------------------
    cx1 = cx0 + cw
    nx = cx1 + 6.2
    node("x_img", nx, yc, "×")
    path([(fc_r, yc), (nx - 2.1, yc)])
    edge_label((cx1 + nx - 2.1) / 2, yc - 1.9, "$h_{img}$")
    path([(28.5, 12.0), (28.5, 4.8), (nx, 4.8), (nx, yc - 2.1)], kind="q")
    tx(29.7, 7.0, "$q$ (Eq. 3)", size=S, ha="left")
    edge_label(nx + 2.3, yc - 3.4, "$Q_{img} = q$", ha="left")

    # structured branch -------------------------------------------------------------
    y2 = 54.0
    box("desc", 2, 48, 24, 12, "Descriptors $d_n$", ["11 values, Table 2", "on $\\tilde x$ (Eq. 1), $M_n$"])
    path([(26, y2), (37, y2)])
    edge_label(31.5, y2 - 2.0, "$d_n \\in \\mathbb{R}^{11}$")
    box("prep", 37, 48, 27, 12, "Preparation", ["impute, z-score → $\\tilde d$", "$Q_{struct}$: Eq. (10)"])
    node("x_struct", 71.0, y2, "×")
    path([(64, y2), (68.9, y2)])
    path([(60.5, 48.0), (60.5, 44.6), (71.0, 44.6), (71.0, y2 - 2.1)], kind="q")
    edge_label(66.0, 43.0, "$Q_{struct}$ (= 1 on TN3K)")
    box("clin", 2, 62.0, 62, 8.6, "Laboratory, clinical, regional inputs",
        ["$\\tilde l_n, \\tilde c_n, \\tilde r_n$ with $Q_{lab}, Q_{clin}, Q_{reg}$; no data in TN3K"],
        ls=DASH, tcolor=GREY)

    # fusion -------------------------------------------------------------------------
    fx0, fy0, fw_, fh = 112.0, 46.0, XMAX - 112.0, 22.0
    box("fusion", fx0, fy0, fw_, fh, "Quality-weighted fusion — Eq. (11)",
        ["$z = [\\,Q_{img}h_{img};\\ Q_{struct}\\tilde d;\\ Q_{img};\\ Q_{struct}\\,]$"], dy=-3.2)
    # layout of z as a segmented vector (widths not to scale)
    segs, sx0, sy0, sh = [("128", 20.0), ("11", 7.5), ("1", 4.0), ("1", 4.0)], fx0 + 2.4, 59.0, 4.0
    for j, (n_, w_) in enumerate(segs):
        rbox(f"zseg{j}", sx0, sy0, w_, sh, radius=0.3, lw=0.6, parent="fusion", z=2.5)
        tx(sx0 + w_ / 2, sy0 + sh / 2, n_, size=S, color=INK, parent=f"zseg{j}", inside=True)
        sx0 += w_
    tx(sx0 + 1.5, sy0 + sh / 2, "= 141 on TN3K", size=L, ha="left", parent="fusion", inside=True)
    path([(nx, yc + 2.1), (nx, fy0)])
    edge_label(nx + 1.2, 33.4, "$Q_{img}h_{img}$", ha="left")
    edge_label(nx + 1.2, 36.2, "$\\in \\mathbb{R}^{128}$", ha="left")
    path([(73.1, y2), (fx0, y2)])
    edge_label(92.5, y2 - 2.0, "$Q_{struct}\\tilde d \\in \\mathbb{R}^{11}$")
    path([(64, 66.3), (fx0, 66.3)], kind="dash")

    # learners -----------------------------------------------------------------------
    zx, ybar, ly0, lh = 140.0, 72.5, 76.5, 12.5
    path([(zx, fy0 + fh), (zx, ybar)], arrow=False)
    dot(zx, ybar)
    edge_label(zx + 1.3, 70.2, "$z \\in \\mathbb{R}^{141}$", ha="left")
    lx0, lw_, lg = 22.0, 34.0, (XMAX - 22.0 - 4 * 34.0) / 3
    learners = [
        ("DL head $g_\\phi$", ["141 → 64 → 1, sigmoid", "ReLU, dropout 0.3",
                               "end-to-end with $f_\\theta$"], dict(fill=FC_FILL, edge=FC_EDGE)),
        ("Random forest", ["500 trees, class-balanced", "$p$ = mean leaf class fraction", "fitted on $z$"], {}),
        ("XGBoost", ["≤ 600 rounds, logistic", "early-stopped on val.", "fitted on $z$"], {}),
        ("Fuzzy SVM", ["RBF on standardized $z$", "$s_n$: Eqs. (12)–(13)", "Platt scaling on val."], {}),
    ]
    cxs = []
    for i, (t, ls_, kw) in enumerate(learners):
        x = lx0 + i * (lw_ + lg)
        cxs.append(x + lw_ / 2)
        box(f"learner{i}", x, ly0, lw_, lh, t, ls_, **kw)
    AX.plot([cxs[0], cxs[-1]], [ybar, ybar], color=RULE, lw=LW, solid_capstyle="butt", zorder=1)
    for c in cxs:
        path([(c, ybar), (c, ly0)])
    # SHAP tap on z
    box("shap", 2, ly0, 17, lh, "SHAP", ["on $z$:", "TreeSHAP,", "KernelSHAP"], ls=DOT, edge=GREY)
    path([(cxs[0], ybar), (10.5, ybar), (10.5, ly0)], kind="tap")

    # ensemble, prior shift, seeds, decision, output ----------------------------------
    y4, ycol = 102.5, 92.5
    for c in cxs:
        AX.plot([c, c], [ly0 + lh, ycol], color=RULE, lw=LW, solid_capstyle="butt", zorder=1)
    sx = 30.0
    AX.plot([sx, cxs[-1]], [ycol, ycol], color=RULE, lw=LW, solid_capstyle="butt", zorder=1)
    edge_label((cxs[0] + cxs[1]) / 2, 90.8, "$p_k \\in [0,1]$, $k$ = 1…4")
    node("sigma_sum", sx, y4, "Σ", r=2.6)
    path([(sx, ycol), (sx, y4 - 2.6)])
    tx(26.4, y4 - 3.3, "Eq. (14)", size=L, ha="right")
    tx(26.4, y4 - 0.1, "$p_{ens} = \\Sigma_k\\,\\omega_k\\,p_k$", size=L, ha="right")
    tx(26.4, y4 + 3.2, "$\\omega_k \\propto \\mathrm{exp}(-L_k/\\tau)$", size=L, ha="right")
    box("prior", 40, y4 - 6.5, 44, 13, "Prior shift",
        ["Eq. (15): $\\Delta_{reg} = 0$ on TN3K",
         "Eq. (16): $p_{final} = \\sigma(\\mathrm{logit}\\,p_{ens} + \\Delta_{reg})$"])
    path([(sx + 2.6, y4), (40, y4)])
    edge_label(36.3, y4 - 1.9, "$p_{ens}$")
    box("seeds", 92, y4 - 5.5, 17, 11, "Seed mean", ["seeds 0–4"])
    path([(84, y4), (92, y4)])
    edge_label(88, y4 - 1.9, "$p_{final}$")
    box("decision", 117, y4 - 6.5, 31, 13, "Decision — Eq. (16)",
        ["$\\hat y = 1[\\,\\bar p_{final} \\geq t^{*}]$", "$t^{*}$: sens. ≥ 0.95 (val.)"], lead=1.38)
    path([(109, y4), (117, y4)])
    edge_label(113, y4 - 1.9, "$\\bar p_{final}$")
    box("output", 152, y4 - 6.5, XMAX - 152, 13, "Output", ["$\\hat y \\in \\{0,1\\}$", "$\\bar p_{final}$"],
        fill=TINT, lead=1.38)
    path([(148, y4), (152, y4)])


# ---------------------------------------------------------------------------- panel (b)
def panel_b(maps):
    y0 = 113.0
    sub = "maps of TN3K test image te_0450 after crop and resize to 224²" if maps else "every map is H × W"
    panel_title(2, y0, "(b)", f"Fuzzy front end on the native frame, Eqs. (1)–(8); {sub}")
    ybus, by, bh = y0 + 4.8, y0 + 7.0, 11.0
    ych = by + bh + 2.3                                             # channel below the boxes
    n, g = 8, 2.4
    bw = (XMAX - 2.0 - (n - 1) * g) / n
    cols = [2.0 + i * (bw + g) for i in range(n)]
    cc = [c + bw / 2 for c in cols]
    steps = [
        ("Min–max", ["$\\tilde x \\in [0,1]$", "Eq. (1)"], True),
        ("Local stats", ["$c, \\sigma$; $k$ = 7", "Eq. (2)"], True),
        ("$T, K, q$", ["texture, SNR", "Eq. (3)"], True),
        ("ATCFM $\\mu$", ["Eq. (4); from", "$\\tilde x, c, \\sigma, T, K, q$"], True),
        ("$r, w$", ["from $\\sigma, \\sigma_g$", "Eqs. (5)–(6)"], True),
        ("$\\varepsilon_w, \\delta_w$", ["on $\\mu$ with $r, w$", "Eq. (7)"], True),
        ("AFMO $F$", ["$\\lambda O + (1-\\lambda) C$", "Eq. (8)"], True),
        ("Otsu$(F)$", ["bright side", "vs. $M_n$"], False),
    ]
    for i, (t, ls_, acc) in enumerate(steps):
        kw = dict(fill=ACCENT_FILL, edge=ACCENT, lw=1.0) if acc else {}
        box(f"b{i}", cols[i], by, bw, bh, t, ls_, **kw)
    ym = by + bh / 2
    for a_, b_ in ((0, 1), (2, 3), (4, 5), (5, 6), (6, 7)):          # adjacent dependencies
        path([(cols[a_] + bw, ym), (cols[b_], ym)])
    # c, σ → μ and σ → r, w (bus above the boxes)
    path([(cc[1], by), (cc[1], ybus), (cc[4], ybus), (cc[4], by)])
    path([(cc[3] - 3.0, ybus), (cc[3] - 3.0, by)])
    dot(cc[3] - 3.0, ybus)
    edge_label((cc[1] + cc[3]) / 2, ybus - 1.5, "$c, \\sigma$")
    edge_label((cc[3] + cc[4]) / 2 + 1.0, ybus - 1.5, "$\\sigma$")
    # x̃ → T, K, q and μ → ε_w, δ_w (channel below the boxes)
    path([(cc[0] + 4.0, by + bh), (cc[0] + 4.0, ych), (cc[2] - 4.0, ych), (cc[2] - 4.0, by + bh)])
    path([(cc[3] + 4.0, by + bh), (cc[3] + 4.0, ych), (cc[5] - 4.0, ych), (cc[5] - 4.0, by + bh)])
    # q leaves at Eq. (3): to the membership (above) and to Q_img in (a)
    ty, ts = ych + 3.0, 15.0
    qx = cc[2] + 4.0
    path([(qx, by + bh), (qx, ty + 5.0)], kind="q")
    if maps:
        tx(cc[2], ty + 7.6, f"$q$ = {maps['q']:.2f}", size=L, color=INK)
    tx(cc[2], ty + 10.4, "= $Q_{img}$ in (a)", size=S)
    # outputs to (a)
    yr, yp = ty + ts + 1.6, ty + ts + 4.2
    if maps:
        spec = [(0, "x", "gray", "$\\tilde x$"), (1, "sigma", "magma", "$\\sigma$"), (3, "mu", "magma", "$\\mu$"),
                (4, "r", "magma", "$r$"), (6, "F", "magma", "$F$")]
        for i, k, cm, sym in spec:
            x0 = cc[i] - ts / 2
            lo, hi = thumb(f"th_{k}", x0, ty, ts, maps[k], cm)
            rng = f"{lo:.0f}–{hi:.0f}" if k == "r" else f"{lo:.2f}–{hi:.2f}"
            tx(cc[i], yr, f"{sym}: {rng}", size=S)
        thumb("th_seg", cc[7] - ts / 2, ty, ts, maps["seg"], "gray", contour=maps["gt"], vmin=0, vmax=1)
        tx(cc[7], yr, f"Dice {maps['dice']:.2f} vs. $M_n$", size=S)
    tx(cc[5], ty + 6.0, "$\\lambda$ = 0 selected:", size=S)
    tx(cc[5], ty + 8.6, "$F = C$ (closing)", size=S)
    tx(cc[0], yp, "channel 1 of (a)", size=S)
    tx(cc[6], yp, "channel 2 of (a)", size=S)
    tx(cc[7], yp, "E1: crop, $d_n$", size=S)
    return yp


# ---------------------------------------------------------------------------- legend
def legend(y):
    def sw_box(x, yy, **kw):
        AX.add_patch(FancyBboxPatch((x, yy - 1.4), 4.4, 2.8, boxstyle="round,pad=0,rounding_size=0.5",
                                    zorder=2, **kw))
        return x + 5.6

    def sw_line(x, yy, kind, arrow=True):
        path([(x, yy), (x + 6.0, yy)], kind=kind, arrow=arrow)
        return x + 7.2

    items1 = [(lambda x: sw_box(x, y, facecolor=ACCENT_FILL, edgecolor=ACCENT, linewidth=1.0),
               "fuzzy front end (this work)"),
              (lambda x: sw_box(x, y, facecolor=CONV_FILL, edgecolor=CONV_EDGE, linewidth=LW), "convolution"),
              (lambda x: sw_box(x, y, facecolor=FC_FILL, edgecolor=FC_EDGE, linewidth=LW), "dense / pooling"),
              (lambda x: sw_box(x, y, facecolor="white", edgecolor=RULE, linewidth=LW), "module"),
              (lambda x: sw_box(x, y, facecolor=TINT, edgecolor=RULE, linewidth=LW), "input / output"),
              (lambda x: _sw_node(x, y), "operator")]
    y2 = y + 3.8
    items2 = [(lambda x: sw_line(x, y2, "data"), "data flow"),
              (lambda x: sw_line(x, y2, "q"), "quality coefficient $q$, $Q_{struct}$"),
              (lambda x: sw_box(x, y2, facecolor="white", edgecolor=RULE, linewidth=LW, linestyle=DASH),
               "framework-defined, no data in TN3K"),
              (lambda x: sw_line(x, y2, "tap"), "evaluation-time tap"),
              (lambda x: _sw_contour(x, y2), "$M_n$ contour in (b)")]
    for items, yy in ((items1, y), (items2, y2)):
        x = 2.0
        for draw, text in items:
            x = draw(x)
            t = tx(x, yy, text, size=L, ha="left", color=INK)
            FIG.canvas.draw()
            bb = t.get_window_extent(FIG.canvas.get_renderer()).transformed(AX.transData.inverted())
            x = max(bb.x0, bb.x1) + 3.4


def _sw_contour(x, y):
    AX.plot([x, x + 6.0], [y, y], color=MASK_C, lw=0.9, solid_capstyle="butt", zorder=2)
    return x + 7.2


def _sw_node(x, y):
    AX.add_patch(Circle((x + 1.6, y), 1.6, facecolor="white", edgecolor=INK, linewidth=LW, zorder=3))
    AX.text(x + 1.6, y - 0.05, "×", ha="center", va="center", fontsize=6, color=INK, zorder=4)
    return x + 4.4


# ---------------------------------------------------------------------------- layout check
def check_layout():
    """Report text outside its box, text crossing other boxes, text–text overlaps, small type and
    anything beyond x = 168 mm. Coordinates in mm."""
    FIG.canvas.draw()
    r = FIG.canvas.get_renderer()
    inv = AX.transData.inverted()
    items, problems = [], []
    for t, parent, inside in G.texts:
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        (ax0, ay0), (ax1, ay1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        x0, x1, y0, y1 = min(ax0, ax1), max(ax0, ax1), min(ay0, ay1), max(ay0, ay1)
        items.append((t, x0, y0, x1, y1, parent))
        name = repr(t.get_text())
        if t.get_fontsize() < 5.5 - 1e-6:
            problems.append(f"size {t.get_fontsize()} pt < 5.5: {name}")
        if x1 > XMAX + 1e-6 or x0 < 0.6:
            problems.append(f"outside 0.6–{XMAX} mm (x {x0:.2f}–{x1:.2f}): {name}")
        if inside and parent:
            px0, py0, px1, py1, _ = G.rects[parent]
            pad = 0.35
            if x0 < px0 + pad or x1 > px1 - pad or y0 < py0 + 0.2 or y1 > py1 - 0.2:
                problems.append(f"overflows box {parent}: {name} x {x0:.2f}–{x1:.2f} / box {px0:.2f}–{px1:.2f};"
                                f" y {y0:.2f}–{y1:.2f} / box {py0:.2f}–{py1:.2f}")
        anc = G.ancestors(parent) if parent else set()
        for k, (rx0, ry0, rx1, ry1, _) in G.rects.items():
            if k in anc:
                continue
            if x0 < rx1 - 0.05 and x1 > rx0 + 0.05 and y0 < ry1 - 0.05 and y1 > ry0 + 0.05:
                problems.append(f"text crosses box {k}: {name}")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a[1] < b[3] - 0.05 and a[3] > b[1] + 0.05 and a[2] < b[4] - 0.05 and a[4] > b[2] + 0.05:
                problems.append(f"text overlap: {a[0].get_text()!r} × {b[0].get_text()!r}")
    for k, (rx0, ry0, rx1, ry1, _) in G.rects.items():
        if rx1 > XMAX + 1e-6:
            problems.append(f"box {k} ends at x = {rx1:.2f} > {XMAX}")
    margins = []
    for t, x0, y0, x1, y1, parent in items:
        if parent and parent in G.rects:
            px0, _, px1, _, _ = G.rects[parent]
            margins.append((min(x0 - px0, px1 - x1), t.get_text(), parent))
    margins.sort()
    print(f"[fig2] {len(items)} texts, {len(G.rects)} boxes; tightest horizontal margins (mm):")
    for m, s, p in margins[:6]:
        print(f"         {m:5.2f}  {p:10s} {s!r}")
    sizes = sorted({round(t.get_fontsize(), 2) for t, *_ in items})
    print(f"[fig2] font sizes used (pt): {sizes}")
    print("[fig2] layout check: " + ("clean" if not problems else f"{len(problems)} problem(s)"))
    for p in problems:
        print("   - " + p)
    return problems


def main():
    global AX, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    ap.add_argument("--no-maps", action="store_true", help="draw panel (b) without the example maps")
    a = ap.parse_args()
    fam = setup_font()
    FIG, AX = canvas(W, H)
    maps = None if a.no_maps else example_maps()
    panel_a()
    yb = panel_b(maps)
    legend(yb + 4.6)
    print(f"[fig2] font: {fam}")
    check_layout()
    print(save(FIG, "fig2_architecture", a.out))


if __name__ == "__main__":
    main()
