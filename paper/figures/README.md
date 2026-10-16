# Figures

All figures are produced by `scripts/plots/*.py` from saved data only (no retraining). Figures 1 and 2 (diagrams)
use `scripts/plots/figstyle_diagram.py` (`setup_font()` picks Helvetica on this Mac; both scripts accept `--out`);
Figures 3 to 8 and S1 share `scripts/plots/_style.py`: Okabe–Ito palette with one fixed colour and line style per model in every figure
(TAMES AI blue solid; image-only CNN orange dashed; late fusion green dash-dot; descriptor-only XGBoost grey
dotted; TAMES AI with Q_m = 1 purple thin; single learners light grey thin), Helvetica/Arial, 8 pt axes, 7 pt
ticks and legends, 9 pt bold panel letters, every symbol in mathtext, PDF (vector) + PNG at 600 dpi cropped to
content (`bbox_inches='tight'`, pad 0.02 in), widths 85 mm (single column) or 170 mm (double column). Regenerate everything with:

    source .venv/bin/activate
    for f in scripts/plots/fig*.py; do python "$f"; done

| Figure | File | Script | Inputs |
|---|---|---|---|
| 1 | `fig1_flow.{pdf,png}` | `fig1_flow.py` (+ `figstyle_diagram.py`) | `data/processed/splits.csv` |
| 2 | `fig2_architecture.{pdf,png}` | `fig2_architecture.py` (+ `figstyle_diagram.py`) | (a) data-flow graph with tensor shapes; (b) fuzzy front end with the real maps of test image te_0450 (raw image from `data/raw/tn3k`, SNR₀ from `data/processed/snr0.json`, α 2, β 0.5, γ 2, λ 0, bright polarity); notes in `notes/fig2_v3_NOTES.md` |
| 3 | `fig3_samples.{pdf,png}` (170 mm) | `fig3_samples.py` | `data/raw/tn3k` images and masks, `data/processed/{splits.csv, meta_test.csv, X_test.npy, fuzzy_params.json}`, `runs/tn3k_fold0/ensemble_tames_seed0/{p_pooled_test.npy, model_order.json, results.json}`; the whole-image maps μ and F and Otsu(F) are recomputed from the raw image with the saved parameters |
| 4 | `fig4_roc.{pdf,png}` (170 mm) | `fig4_roc.py` | pooled test predictions and `results.json` of `runs/tn3k_fold0` (a) and `runs/tn3k_automask` (b); labels from `meta_test.csv` |
| 5 | `fig5_robustness.{pdf,png}` (170 mm) | `fig5_robustness.py` | `runs/tn3k_fold0/robustness/robustness.csv` |
| 6 | `fig6_lowdata.{pdf,png}` (85 mm) | `fig6_lowdata.py` | `runs/lowdata_table.csv` |
| 7 | `fig7_calibration.{pdf,png}` (170 mm) | `fig7_calibration.py` | `results.json → models.*.calibration_test.bins` (a; point area 10–120 pt² for n = 30–90); `p_pooled_test.npy` (b) |
| 8 | `fig8_explanations.{pdf,png}` (170 mm) | `fig8_explanations.py` | `runs/tn3k_fold0/explain_tames_seed0/{shap_summary.csv, shap_stability.json}`, seed-0 weights from `results.json` (a); `cnn_tames_seed0/model.pt`, `X_test.npy`, the `GradCAM` class of `scripts/07_explain.py` (b, inference only) |
| S1 | `figS1_layers.{pdf,png}` | `figS1_layers.py` (+ `figstyle_diagram.py`) | none; layered view of the architecture (eight layers), key: evaluated / defined / contribution |
| S1 (old numbering, see note) | `figS1_seed_roc_confusion.{pdf,png}` (170 mm) | `figS_supplement.py` | per-seed `p_test.npy`, `p_{rf,xgb,fsvm}_test.npy` and per-seed weights (a); pooled predictions and thresholds (b) |

Notes.
* Figure 3 rows: three benign (two true negatives, one false positive) above three malignant (two true positives,
  one false negative) at t*, drawn with `numpy.random.default_rng(0)`; every panel is a square (the whole image is
  padded with black); μ and F are shown with a per-map 1–99th-percentile display stretch (magma; state "display
  range 1–99th percentile" in the caption); column (v) shows the raw Otsu(F) foreground as a translucent blue fill with the expert delineation as a
  dashed white outline, without post-processing, so the panel shows exactly what the Dice in Section 5.2 was computed on
  (caption: "(v) foreground of the Otsu threshold of F over the crop, expert delineation dashed; Dice in Section 5.2
  is computed on the whole image"); column (i) pads the image to square with white and frames the actual image extent.
* Figure 8a: x-axis scaled to the descriptors (0 to 1.1 × the largest descriptor value); the image-embedding bar
  is hatched, clipped at the axis edge and labelled with its value (0.278, 84.9 % of the total); the other bars are
  split into ω_k·mean|φ_k| per learner, rescaled so the parts sum to the ensemble mean |SHAP| of that feature; the
  percentage is the share of 2,000 bootstrap resamples in which the feature stayed in the top ten (top-10 only).
* Figure 8b: rows TP, TN, FP, FN, one case each (the first of the `default_rng(0)` pairs); tiles image | Grad-CAM
  normalized to the global maximum over the 32 images audited by step 7 | Grad-CAM normalized to its own maximum
  (caption note: "left overlay: global scale; right overlay: per-map scale"); overlay opacity proportional to
  saliency, so weak maps stay grey on the global scale; raw maps from the seed-0 backbone.
* Figures 4, 5, 6 and S1 use the 95 % bootstrap CIs stored by the pipeline; nothing is re-estimated.

* Figure captions for Figure 2 (computational graph) and Supplementary Figure S1 (layered view) follow the master document; they are added here when the manuscript zip with the new captions arrives.
* Numbering clash: `figS1_seed_roc_confusion` (per-seed ROC and confusion matrices) predates the layered view and also carries S1; it needs a new number (e.g. S2) once the supplementary figure order is fixed.
