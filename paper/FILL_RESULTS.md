# How the results get into the paper

`manuscript_v2.md` is the full revised manuscript. Every `[[...]]` is a value that must come
from the experiment or from the authors. After `bash scripts/run_all.sh` finishes, fill them
from these files (all under `runs/tn3k_fold0/`):

| Manuscript location | Source |
|---|---|
| Abstract › Methods: N, n, institution, period | `data/processed/splits.csv` (counts per split); institution/period = TN3K source (Zhujiang Hospital, Southern Medical University; see Gong et al. 2021) |
| Abstract › Results; 5.3 Table 3 | `ensemble_tames_seed0/results_table.csv`; CIs and the DeLong/McNemar/Holm statistics in `results.json → models`, `comparisons_vs_tames` |
| 5.1 Cohort, Table 1 | `data/processed/splits.csv`, `meta_*.csv` (label fractions, image sizes H/W); scanners = "various devices" per the TN3K release |
| 5.2 Segmentation (λ, Dice) | `data/processed/fuzzy_params.json` (α, β, γ, λ, polarity, val Dice, raw-Otsu Dice); test Dice = mean of `dice_afmo` / `dice_raw` in `meta_test.csv` |
| 5.4 External validation | not available until a DDTI loader is added (README §7) — mark the paragraph "not performed" or remove |
| 5.5 Calibration | `results.json → models.tames.calibration_test` (brier, slope, intercept, bins) |
| 5.6 Table 4, ensemble weights, τ | `ablation_table.csv`; `results.json → per_seed[*].weights, tau` |
| 5.7 Subgroups | TN3K has no Bethesda/sex/scanner metadata: keep only the size subgroup (compute from `area_frac` or drop the table) |
| 5.8 Explanations, Figure 5 | `explain_tames_seed0/shap_summary.csv`, `shap_stability.json`, `faithfulness.json`, `shap_summary.png`, `gradcam.png` |
| 3.13 Implementation | package versions (`pip freeze`), hardware, training time from `cnn_tames_seed*/cnn_results.json → history` |
| Supplementary S2 | `data/processed/fuzzy_params.json`, `learners_tames_seed*/learner_params.json` |

Rules for whoever fills it (human or Claude Code):
1. Copy numbers exactly as printed; never round in a way that changes the first decimal of an AUC.
2. If a value does not exist in the outputs, leave the `[[...]]` in place — do not invent it.
3. Baseline rows that were not run (TI-RADS alone) are deleted from Table 3, with one sentence saying why.
4. The abstract's prevalence, sample counts and AUCs must equal those in Section 5.
5. After filling, search the file for `[[` — the remaining items are the authors' (ethics, affiliations, data statement).
