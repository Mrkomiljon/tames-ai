"""Shared drawing helpers for the two diagram figures (fig1_flow, fig2_architecture).

Coordinates are millimetres on an axes with equal aspect; the figure size is set from the
drawing extent, so widths come out exactly as 85 mm / 170 mm. Font: Helvetica if present,
else Arial, Liberation Sans, Inter, DejaVu Sans (mathtext follows the same family).
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Okabe–Ito, as in _style.py
INK = "#1b1d1f"
GREY = "#5d6166"
RULE = "#8a8f94"
ACCENT = "#0072B2"
ACCENT_FILL = "#e6f0f8"
TINT = "#f2f2ef"
TEST_FILL = "#eaf3fb"


def setup_font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("Helvetica", "Helvetica Neue", "Arial", "Liberation Sans", "Inter", "DejaVu Sans"):
        if fam in names:
            break
    matplotlib.rcParams.update({
        "font.family": fam, "font.size": 8,
        "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": f"{fam}:italic",
        "mathtext.bf": f"{fam}:bold", "mathtext.default": "it",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    return fam


def canvas(width_mm: float, height_mm: float):
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_mm); ax.set_ylim(height_mm, 0)   # y grows downward, like a page
    ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title=None, lines=(), *, fill="white", edge=RULE, lw=0.8, ls="-",
        title_size=8, line_size=6.8, title_color=INK, line_color=GREY, radius=1.2, align="center",
        title_weight="bold", gap=3.0):
    """Rounded box with a bold title line and grey body lines, vertically centred."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=fill, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=2))
    n = (1 if title else 0) + len(lines)
    lead_t, lead_l = 3.6, 2.9
    total = (lead_t if title else 0) + lead_l * len(lines) - (0.7 if n else 0)
    cy = y + h / 2 - total / 2
    tx = x + w / 2 if align == "center" else x + 2.5
    ha = "center" if align == "center" else "left"
    if title:
        cy += lead_t / 2
        ax.text(tx, cy, title, ha=ha, va="center", fontsize=title_size, color=title_color,
                fontweight=title_weight, zorder=3)
        cy += lead_t / 2 + gap * 0.25
    for ln in lines:
        cy += lead_l / 2
        ax.text(tx, cy, ln, ha=ha, va="center", fontsize=line_size, color=line_color, zorder=3)
        cy += lead_l / 2
    return (x, y, w, h)


def arrow(ax, p0, p1, *, elbow=None, color=RULE, lw=0.8, head=5):
    """Arrow from p0 to p1; elbow='v' goes vertical then horizontal, 'h' horizontal then vertical."""
    style = f"-|>,head_length={head * 0.6},head_width={head * 0.35}"
    if elbow is None:
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color, linewidth=lw,
                                     mutation_scale=1, shrinkA=0, shrinkB=0, zorder=1))
        return
    mid = (p0[0], p1[1]) if elbow == "v" else (p1[0], p0[1])
    ax.plot([p0[0], mid[0]], [p0[1], mid[1]], color=color, lw=lw, solid_capstyle="round", zorder=1)
    ax.add_patch(FancyArrowPatch(mid, p1, arrowstyle=style, color=color, linewidth=lw,
                                 mutation_scale=1, shrinkA=0, shrinkB=0, zorder=1))


def label(ax, x, y, text, *, size=7.2, color=GREY, ha="right", weight="normal", style="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color, fontweight=weight,
            fontstyle=style, zorder=3)


def save(fig, stem, outdir):
    from pathlib import Path
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    return out / f"{stem}.png"
