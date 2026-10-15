# Remaining items for paper/manuscript_v3.md (pretrained run, commit 86e51cf)

manuscript_v3.md is the export of the master Claude doc after the pretrained results were
written in. All tables (3–7) and all AUC/p-values are final. The 46 remaining [[...]] are:

## A. Read from the run outputs (fill exactly as printed)
| Placeholder | Source |
|---|---|
| Table 1 native image size `[[H range]] × [[W range]]`; 5.1 `[[H×W min]]`, `[[H×W max]]` | `data/processed/splits.csv` columns H, W (min/max) |
| 5.1 `[[k]]` empty delineations | `meta_*.csv` → sum of `mask_empty` over the three splits |
| 5.1 SNR₀ `[[ ]]` | `data/processed/snr0.json` |
| 5.2 Wilcoxon `[[< 0.001]]` | paired Wilcoxon of `dice_afmo` vs `dice_raw` in `meta_test.csv` (scipy.stats.wilcoxon); write the exact p |
| 5.3 McNemar `p = [[ ]] and [[ ]]` | `runs/tn3k_fold0/ensemble_tames_seed0/results.json → comparisons_vs_tames.cnn_image_only.mcnemar_sensitivity/specificity.p` |
| 5.8 Youden `[[ ]] % and [[ ]] %` | `results.json → models.tames.test_youden.sensitivity/specificity` |
| 5.9 SHAP share `[[ ]] of the total` | `runs/tn3k_fold0/explain_tames_seed0/shap_summary.csv`: ensemble column, image-embedding row ÷ column sum |
| 5.9 size subgroup `[[ ]] (below) / [[ ]] (above)` | split test rows by median `area_frac` in `meta_test.csv`; AUC of the pooled tames predictions (`ensemble_tames_seed0/p_pooled_test.npy`, row order in `model_order.json`) in each half |
| 3.12 sample size `[[ ]]`, `[[n]]`, `[[e]]`, `[[e_train]]`, `[[meets / does not meet]]` | e_train = malignant count in train split; apply Riley et al. 2020 criteria with AUC 0.80 anticipated, prevalence 0.338, 141 predictors (pmsampsize R package or its formulas) |
| 3.13 package versions `[[x.y]]` ×5, hardware, training time `[[min]]` per seed | `pip freeze`; machine spec; `cnn_tames_seed*/cnn_results.json → history` length × epoch time from the log |
| 4 inference time `[[x]] s per image on [[hardware]]` | time `fuzzy_maps` + CNN forward + learners on 100 test images |

## B. Author items
`[[name, e-mail, ORCID]]`, `[[public repository URL, commit hash]]`, `[[DOI]]`, `[[none / grant number]]`,
author contributions, and the `[[verify ...]]` reference items (12, 14, 15, 18, 20–22: complete the
bibliographic data from the publisher pages).

## C. Figures to produce (scripts to write under scripts/plots/, from saved predictions only)
- Figure 1: image flow (3,493 → 2,303 / 576 / 614), a simple box diagram.
- Figure 3 (v3 numbering): dataset samples and fuzzy pipeline; μ and F are shown with a per-map 1–99th-percentile display stretch (state "display range 1–99th percentile" in the caption).
- Figure 8b (v3 numbering): Grad-CAM gallery, caption note "left overlay: global scale; right overlay: per-map scale".
- Figure 4 (v3 numbering): ROC curves of all Table-3 models on the test set, AUC + CI in the legend
  (`p_pooled_test.npy`, `model_order.json`, labels from `meta_test.csv`).
- Figure 4: `runs/tn3k_fold0/robustness/robustness.png` (exists).
- Figure 5: `runs/lowdata.png` (exists).
- Figure 6: calibration curves (10 quantile bins) for tames, cnn_image_only, late_fusion (`results.json → calibration_test.bins`).
- Figure 7: `explain_tames_seed0/shap_summary.png` and `gradcam.png` (exist) as panels a and b.
- Supplementary S1 (TRIPOD+AI, CLAIM checklists), S2 (hyperparameters from `fuzzy_params.json`,
  `learner_params.json`, config), S3 (logistic stacker — not run; delete the sentence in 3.8 or run it),
  S4 (seed-to-seed AUC SD per model from `results.json`).

Rules unchanged: copy numbers exactly; leave anything not found as [[...]]; never alter Tables 3–7.
