"""Shared figure style: validated categorical palette (fixed slot per model), thin marks, recessive axes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed slot per model (color follows the entity across all figures). Palette validated for CVD separation.
SLOT = {"tames": "#2a78d6", "dl_only": "#eb6834", "rf_only": "#1baf7a", "xgb_only": "#eda100", "fsvm_only": "#e87ba4",
        "xgb_struct_only": "#008300", "cnn_image_only": "#6250d6", "late_fusion_cnn_xgb": "#e34948"}
LABEL = {"tames": "TAMES AI", "dl_only": "DL learner alone", "rf_only": "RF learner alone", "xgb_only": "XGB learner alone",
         "fsvm_only": "FSVM learner alone", "xgb_struct_only": "Descriptor-only XGBoost", "cnn_image_only": "Image-only CNN",
         "late_fusion_cnn_xgb": "CNN + XGBoost late fusion"}
ORDER = ["tames", "cnn_image_only", "late_fusion_cnn_xgb", "xgb_struct_only", "dl_only", "rf_only", "xgb_only", "fsvm_only"]
# Secondary encoding (not color alone): full model solid, baselines dashed, single learners dotted.
STYLE = {"tames": ("-", 2.4), "cnn_image_only": ("--", 1.6), "late_fusion_cnn_xgb": ("--", 1.6), "xgb_struct_only": ("--", 1.6),
         "dl_only": (":", 1.4), "rf_only": (":", 1.4), "xgb_only": (":", 1.4), "fsvm_only": (":", 1.4)}
INK, INK2, GRID, SURFACE = "#1f1f1e", "#5c5b57", "#e6e5e1", "#fcfcfb"

def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "savefig.dpi": 200})

def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"paper/figures/{stem}.{ext}", bbox_inches="tight")
    print("wrote", f"paper/figures/{stem}.png")
