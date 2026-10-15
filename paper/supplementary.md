
TAMES AI: An Adaptive Fuzzy, Multimodal Ensemble Model for Malignancy Risk Assessment of Thyroid Nodules — Supplementary Tables S1–S3. Section numbers refer to the main text.

# S1. Reporting checklists

**Table S1a.** TRIPOD+AI items (Collins et al., 2024) and where each is addressed.

| Item | Where addressed |
| ---------------------------- | ---------------------------- |
| Title identifies a prediction-model study, the target population and the outcome | Title |
| Structured abstract: objectives, data, methods, results, conclusions | Abstract |
| Background, rationale and existing models | 1, 2 |
| Objectives: development and evaluation | 1 (last paragraph), 3.1 |
| Source of data, dates, setting | 3.2, Table 1 |
| Eligibility of participants / images | 3.1, 3.2 (every image of the official release, none excluded) |
| Data preparation and pre-processing | 3.4–3.6, 3.13 |
| Outcome definition and how it was assessed | 3.1 (dataset label; cytological basis not documented, 6) |
| Predictors: definition, measurement, pre-processing | 3.2 and Table 2, 3.6 |
| Sample size | 3.12 |
| Missing data | 3.6 (quality coefficients Q\_m); none on TN3K (5.1) |
| Analytical methods: model type, predictor handling, hyperparameter tuning, internal validation | 3.4–3.10, 3.13, Table S2 |
| Class imbalance | 3.13 (class-balanced loss), 3.10 (sensitivity-fixed threshold) |
| Fairness | not assessable: TN3K carries no demographic variables (6) |
| Model output | 3.10 (probability and label), 4 |
| Separation of training and evaluation data | 3.2, Figure 1 |
| Ethical approval | Declarations |
| Funding and competing interests | Declarations |
| Protocol and registration | 3.12 (analysis plan fixed before test scoring and released with the code); not registered |
| Data and code sharing | Declarations, 3.13 |
| Patient and public involvement | none (public dataset) |
| Flow and characteristics of the data | 5.1, Table 1, Figure 1 |
| Model development and full specification | 3.4–3.10, 5.2, 5.3 (ensemble weights), Table S2 |
| Model performance with uncertainty | 5.3–5.9, Tables 3–7, Figures 4–8, Table S3 |
| Model updating | 3.9 (prior shift, not applied here) |
| Interpretation, limitations, usability, next steps | 6, 7 |

**Table S1b.** CLAIM items (Mongan et al., 2020) and where each is addressed.

| Item group | Where addressed |
| ---------------------------- | ---------------------------- |
| Study design: intended use, objectives, retrospective design | 1, 3.1 |
| Data: source, eligibility, pre-processing, de-identification | 3.2, 3.4–3.6, Declarations |
| Data partitions: level of split (image), assignment, sizes | 3.2, Table 1, Figure 1 |
| Reference standard: definition, source, rationale | 3.1 |
| Model: architecture, inputs, outputs, software, initialization, hyperparameters | 3.4–3.10, 3.13, Table S2 |
| Training: augmentation, loss, optimizer, early stopping, final-model selection | 3.13 |
| Evaluation: metrics, uncertainty, robustness, explainability | 3.11, 3.12, 5 |
| Flow diagram of the data | Figure 1 |
| Test-set performance with comparators and confidence intervals | Tables 3–7, Figures 4–7 |
| Failure analysis | 5.9 (Grad-CAM cases, size subgroup), Figure 8 |
| Limitations and implications | 6, 7 |
| Registration, protocol, data and code availability | 3.12, Declarations |

# S2. Hyperparameter grids and selected values

**Table S2.** Every value was selected on the validation partition; the test partition played no part. Grids are those of configs/default.yaml in the released code.

| Component | Hyperparameter | Grid or fixed value | Selection criterion | Selected |
| ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| Fuzzy layer (Eqs. 1–8) | α | {0, 0.25, 0.5, 1, 2} | Dice of Otsu(F) against the dataset masks on 150 validation images | 2 |
|  | β | {0, 0.25, 0.5, 1, 2} | same | 0.5 |
|  | γ | {0.5, 1, 2} | same | 2 |
|  | λ | {0, 0.1, …, 1.0} | same | 0 |
|  | k, r₀, r\_min, r\_max, ε, L | 7, 2, 1, 5, 10⁻⁶, 256 | fixed | — |
| Backbone f\_θ and DL head g\_φ | architecture, initialization | ResNet-18, ImageNet-1k; d\_h = 128; dropout 0.3 | fixed | — |
|  | optimizer | AdamW, lr 3×10⁻⁴, weight decay 10⁻⁴, batch 32, cosine schedule, ≤ 30 epochs | fixed | — |
|  | stopping | early stopping on validation AUC, patience 6 | validation AUC | epoch of best validation AUC per seed |
| Random forest | n\_estimators | 500 | fixed | — |
|  | max\_features | {sqrt, 0.3} | validation AUC | 0.3 / 0.3 / 0.3 / 0.3 / sqrt |
|  | min\_samples\_leaf | {1, 3, 5} | validation AUC | 1 / 5 / 3 / 3 / 1 |
| XGBoost | n\_estimators | ≤ 600 with early stopping (30 rounds) on validation | validation log-loss (early stopping) | 51 / 17 / 55 / 62 / 32 |
|  | max\_depth | {3, 4, 6} | validation AUC | 6 / 3 / 6 / 4 / 3 |
|  | learning\_rate | {0.03, 0.1} | validation AUC | 0.03 / 0.1 / 0.03 / 0.03 / 0.03 |
|  | subsample / colsample\_bytree | 0.8 / {0.6, 0.8} | validation AUC | subsample 0.8 (all seeds); colsample 0.6 / 0.8 / 0.8 / 0.8 / 0.8 |
| Fuzzy SVM (Eqs. 12–13) | C | {0.3, 1, 3, 10} | validation AUC of the decision value | 10 / 10 / 10 / 10 / 3 |
|  | γ (RBF) | {scale, 0.01, 0.1} | same | scale / 0.01 / 0.01 / scale / 0.1 |
|  | membership δ; calibration | 10⁻³; Platt scaling fitted on validation | fixed | — |
| Ensemble (Eq. 14) | τ | {0.05, 0.1, 0.2, 0.5, 1.0} | validation log-loss of p\_ens | 0.05 (all seeds) |
| Decision rule (Eq. 16) | t\* | smallest threshold with validation sensitivity ≥ 0.95 | validation sensitivity | 0.134 |

# S3. Seed-level results

**Table S3.** Test AUC of every model by seed (expert delineation and fully automatic settings), with mean and standard deviation across seeds; the pooled values of Tables 3 and 5 are computed on the concatenated predictions of the five seeds.

| Model | Setting | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean | SD |
| --------------------------- | ----------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| TAMES AI | expert | 0.775 | 0.765 | 0.791 | 0.798 | 0.740 | 0.774 | 0.020 |
| Image-only CNN | expert | 0.808 | 0.817 | 0.823 | 0.775 | 0.799 | 0.804 | 0.017 |
| CNN + XGBoost late fusion | expert | 0.782 | 0.800 | 0.779 | 0.773 | 0.795 | 0.786 | 0.010 |
| Descriptor-only XGBoost | expert | 0.678 | 0.687 | 0.680 | 0.685 | 0.683 | 0.682 | 0.003 |
| TAMES AI | automatic | 0.763 | 0.775 | 0.753 | 0.771 | 0.802 | 0.773 | 0.016 |
| Image-only CNN | automatic | 0.766 | 0.783 | 0.767 | 0.798 | 0.767 | 0.776 | 0.013 |
| CNN + XGBoost late fusion | automatic | 0.737 | 0.762 | 0.752 | 0.758 | 0.743 | 0.750 | 0.009 |
| Descriptor-only XGBoost | automatic | 0.577 | 0.587 | 0.588 | 0.575 | 0.583 | 0.582 | 0.005 |
