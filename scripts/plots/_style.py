"""Shared journal style for every figure: Okabe–Ito palette, one fixed colour AND line style per model,
85 / 170 mm widths, Helvetica/Arial, PDF (vector) + PNG 600 dpi, panel letters outside the axes."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
W1, W2 = 85 * MM, 170 * MM
INK, INK2, GRID, TINT = "#1a1a1a", "#555555", "#e4e4e4", "#e8f1fa"
# Okabe–Ito; the same colour + line style identifies a model in every figure.
MODEL = {  # key: (label, colour, linestyle, linewidth)
    "tames": ("TAMES AI", "#0072B2", "-", 1.6),
    "cnn_image_only": ("image-only CNN", "#E69F00", "--", 1.3),
    "late_fusion_cnn_xgb": ("CNN + XGBoost late fusion", "#009E73", "-.", 1.3),
    "xgb_struct_only": ("descriptor-only XGBoost", "#7f7f7f", ":", 1.3),
    "tames_quality_off": ("TAMES AI, Q_m = 1", "#CC79A7", "-", 0.9),
    "dl_only": ("DL learner alone", "#bbbbbb", "-", 0.7),
    "rf_only": ("RF learner alone", "#bbbbbb", "--", 0.7),
    "xgb_only": ("XGB learner alone", "#bbbbbb", "-.", 0.7),
    "fsvm_only": ("FSVM learner alone", "#bbbbbb", ":", 0.7),
}
ROC_ORDER = ["tames", "cnn_image_only", "late_fusion_cnn_xgb", "xgb_struct_only", "dl_only", "rf_only", "xgb_only", "fsvm_only"]

def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False, "legend.handlelength": 2.2,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "lines.linewidth": 1.2, "lines.markersize": 4,
    })

def line(ax, key, x, y, **kw):
    lab, c, ls, lw = MODEL[key]
    return ax.plot(x, y, ls, color=c, lw=lw, label=kw.pop("label", lab), **kw)[0]

def band(ax, key, x, lo, hi):
    ax.fill_between(x, lo, hi, color=MODEL[key][1], alpha=0.15, lw=0)

def panel(ax, letter, dx=-0.14, dy=1.03):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, ha="left", va="bottom")

def save(fig, stem):
    fig.savefig(f"paper/figures/{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"paper/figures/{stem}.png", dpi=600, bbox_inches="tight")
    print("wrote", f"paper/figures/{stem}.{{pdf,png}}")

def results(run="runs/tn3k_fold0"):
    d = f"{run}/ensemble_tames_seed0"
    return json.load(open(f"{d}/results.json")), json.load(open(f"{d}/model_order.json")), d
