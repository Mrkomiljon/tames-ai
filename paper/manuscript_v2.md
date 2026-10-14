# TAMES AI — Revised Manuscript (v2)

Exported from the Claude doc on 2026-10-13. The editable master copy is the Claude doc "TAMES AI — Revised Manuscript (v2)"; keep the two in sync.

## Note to authors (delete before submission)

The manuscript starts at the title below and contains no editorial notes; this block is the only one. All numbers in Section 5 come from the TN3K experiment in the code repository (`configs/default.yaml`, outputs under `runs/tn3k_fold0/`, seeds 0–4); the performance figures of the rejected submission have been withdrawn. Every value still in double brackets, e.g. [[N]], must be supplied by the authors (identities, links, declarations, reference checks) or does not exist in the experiment outputs. Citations are numbered in order of first appearance; references 12, 14, 15 and 18 of the original list are marked [[verify]] because their bibliographic data were incomplete; references 35–37 (TN3K) were appended at the end and must be renumbered into order of first appearance before submission.

## TAMES AI: An Adaptive Fuzzy, Multimodal Ensemble Model for Malignancy Risk Assessment of Thyroid Nodules

**Article type:** Research Article

**Authors:** N. N. Zufarova¹, M. A. Mirzayev²

¹ Muhammad al-Khwarizmi Tashkent University of Information Technologies, Tashkent, Uzbekistan ² Tashkent State University of Economics, Tashkent, Uzbekistan

**Corresponding author:** [[name, e-mail, ORCID]]

### Abstract

**Background.** Malignancy risk stratification of thyroid nodules depends on ultrasound features, laboratory values and cytology that are uncertain, observer-dependent and, in Central Asia, shaped by iodine deficiency. We propose TAMES AI (Thyroid AI Malignancy Evaluation System), a multimodal clinical decision-support framework that represents this uncertainty with adaptive fuzzy set theory and combines it with a hybrid machine-learning ensemble, and we evaluate its image-plus-descriptor instantiation on a public dataset.

**Methods.** The framework was evaluated on the public TN3K dataset: 3,493 B-mode thyroid-nodule images from Zhujiang Hospital, Southern Medical University, with radiologist nodule masks and benign/malignant labels, using the official fold-0 split (2,303 training, 576 validation and 614 test images; image-level, because TN3K has no patient identifiers). Images were processed with an adaptive texture–contrast fuzzy membership function and reliability-weighted fuzzy morphological operations. Image embeddings from a convolutional network were fused with eleven sonographic descriptors computed from the image and the nodule delineation, each modality weighted by a data-quality coefficient; the laboratory, clinical and regional inputs defined by the framework are not available in TN3K and were not evaluated, and the Bayesian prior-shift term for regional prevalence was set to zero. Four learners (deep network, random forest, XGBoost, fuzzy SVM) were combined with validation-loss-dependent weights. Explanations used SHAP and Grad-CAM with faithfulness tests. Performance over five seeds was compared with an image-only CNN, a descriptor-only XGBoost, a CNN + XGBoost late fusion and each learner alone by DeLong and McNemar tests with 95 % bootstrap confidence intervals.

**Results.** On the official TN3K test set (614 images), TAMES AI reached an AUC of 0.750 \[0.710, 0.790\], sensitivity 82.2 % \[77.0, 87.4\], specificity 54.2 % \[49.3, 59.5\] and accuracy 65.0 % at the pre-specified threshold, against an AUC of 0.751 \[0.712, 0.789\] for CNN + XGBoost late fusion (DeLong p = 0.927, Holm-corrected p = 1.000). No external validation was performed. In the ablation, removal of ATCFM (b) cost the most AUC (−0.010 \[−0.030, 0.009\]). The image embedding h\_img (summed over its 128 dimensions), solidity, taller-than-wide ratio, eccentricity, margin irregularity were the dominant SHAP features.

**Conclusions.** The framework represents uncertainty explicitly at the pixel, modality and sample levels. Its image-plus-descriptor instantiation did not improve discrimination over the imaging-only and late-fusion baselines on TN3K; the laboratory, clinical and regional branches and the regional prior are defined but remain to be validated on clinical cohorts. TAMES AI is a research-stage decision-support tool; prospective multicenter evaluation is required before clinical use.

**Keywords:** thyroid nodule; thyroid cancer; ultrasound; adaptive fuzzy set theory; fuzzy morphology; multimodal learning; ensemble learning; explainable artificial intelligence; clinical decision support.

## 1. Introduction

Thyroid nodules are common, and only a small fraction are malignant; the clinical task is to find that fraction without operating on the rest. Current practice relies on ultrasonography graded with the ACR TI-RADS lexicon, laboratory testing, and fine-needle aspiration cytology reported in the Bethesda system \[1–3\]. Each step carries uncertainty. Ultrasound features such as margin irregularity or echogenicity are judged by eye and vary between readers; laboratory values fluctuate; and cytology returns an indeterminate result (Bethesda III–IV) in roughly a fifth of aspirates, a group in which the malignancy rate is too high to ignore and too low to justify surgery for all \[3\]. The consequences are diagnostic surgery for benign disease on one side and delayed cancer diagnosis on the other.

Artificial intelligence has been applied to this problem with promising results. Convolutional networks trained on B-mode images classify nodules with accuracy approaching that of experienced radiologists in meta-analyses \[4–6\]. Three limitations persist. First, most systems use imaging alone and ignore the laboratory and clinical context that the physician has in hand. Second, they treat every input as crisp, although the underlying observations are graded and uncertain; fuzzy set theory offers a mathematical language for exactly this situation \[7\], but its combination with modern multimodal learning remains little explored. Third, models developed on European, North American or East Asian cohorts have not been shown to transfer to Central Asia, where iodine deficiency and regional epidemiology alter both the prevalence and the sonographic appearance of thyroid disease. Explainability adds a fourth requirement: a probability without a rationale is difficult for a clinician to act on, and post-hoc attribution methods such as SHAP and Grad-CAM are now expected in clinical AI reporting \[8, 9\].

This study addresses these limitations with TAMES AI (Thyroid AI Malignancy Evaluation System). The framework (i) represents pixel-level uncertainty with an adaptive texture–contrast fuzzy membership function and reliability-weighted fuzzy morphological operations; (ii) fuses image embeddings with structured variables (laboratory, clinical and regional in clinical deployment; sonographic descriptors in the present experiment), each weighted by an explicit data-quality coefficient; (iii) combines a deep network, a random forest, gradient-boosted trees and a fuzzy support vector machine through validation-loss-dependent ensemble weights; (iv) introduces regional prevalence as a Bayesian prior-shift correction; and (v) delivers SHAP and Grad-CAM explanations whose faithfulness is itself measured.

The aim of this paper is to define the framework completely and to evaluate its image-plus-descriptor instantiation on the public TN3K dataset against single-modality models and a CNN + XGBoost late-fusion baseline under a pre-specified statistical plan; the laboratory, clinical and regional branches are specified for clinical deployment, and their validation on Central Asian cohorts is left to future work. The study is reported according to TRIPOD+AI and CLAIM.

## 2. Related Work and Research Gap

**Imaging-based deep learning.** Convolutional networks \[10\] and residual architectures \[11\] have been trained to classify thyroid nodules from B-mode ultrasound, and encoder–decoder networks derived from U-Net \[12\] to segment them. A model built on ACR TI-RADS descriptors matched radiologist performance in a large single-center series \[13\], and evolutionary encoder–decoder designs have improved nodule segmentation \[14\]. Pooled analyses report AUC values above 0.90 for image-only classifiers \[4–6\], but performance drops when models are applied to other scanners or populations, and a recent two-center external validation documented this gap \[15\].

**Structured machine learning.** Random forests \[16\], gradient-boosted trees \[17\], support vector machines and logistic regression have been applied to clinical, laboratory and sonographic descriptors, and a 2025 meta-analysis found that models built on ultrasound characteristics alone are competitive with image-based networks \[18\]. These methods use the full clinical record but treat ordinal and graded inputs as crisp values. The fuzzy support vector machine \[19\] relaxes this by attaching a membership to every training sample, yet it has rarely been used in thyroid diagnosis.

**Fuzzy and uncertainty-aware approaches.** Fuzzy set theory \[7\] models linguistic categories such as "mildly hypoechoic" directly. Adaptive membership functions, whose parameters follow the local statistics of the data, extend this to image preprocessing. Uncertainty-aware multimodal deep networks have recently been proposed for thyroid ultrasound \[20\], but a model that carries uncertainty from the pixel level through feature fusion to the final decision has not been reported.

**Multimodal fusion.** Combining deep image features with clinical variables improves prediction over either source alone \[21, 22\]. Most published fusions are late fusions of two models; few weight each modality by its measured quality or handle missing laboratory values explicitly.

**Explainability and reporting.** SHAP \[8\] and Grad-CAM \[9\] are the dominant attribution methods in clinical AI; regulators expect them in software-as-a-medical-device submissions \[23\], and reporting guidelines for prediction models \[24\] and imaging AI \[25\] now require that explanation quality be evaluated rather than assumed.

| Approach family | Inputs | Uncertainty handling | Regional adaptation | Explainability | Representative work |
| --- | --- | --- | --- | --- | --- |
| Image-only CNN | ultrasound | none | none | Grad-CAM, sometimes | \[4–6, 11, 13\] |
| Structured ML | clinical, laboratory, descriptors | none | none | feature importance | \[16–18\] |
| Fuzzy SVM | structured | sample membership | none | none | \[19\] |
| Late multimodal fusion | image + clinical | none | none | SHAP on tabular part | \[21, 22\] |
| Uncertainty-aware multimodal DL | image + clinical | evidential or Bayesian output | none | attention maps | \[20\] |
| TAMES AI (this work) | image + sonographic descriptors (evaluated); laboratory, clinical, regional (defined, not evaluated) | pixel membership, quality coefficients, sample membership, prior shift | iodine index, regional prevalence (defined, not evaluated) | SHAP + Grad-CAM with faithfulness tests | — |

**Research gap.** No published model combines adaptive fuzzy image processing, quality-weighted multimodal fusion, an uncertainty-aware ensemble and regional epidemiological correction in one architecture, and none has been developed and externally validated on Central Asian data. TAMES AI is designed to fill that gap.

## 3. Materials and Methods

### 3.1 Study design and reporting

This is a retrospective diagnostic model development study with internal testing on a public dataset, reported according to TRIPOD+AI \[24\] and CLAIM \[25\]; completed checklists are given in Supplementary S1. The unit of analysis is the thyroid-nodule image. Data come from the public TN3K dataset \[35, 36\]: B-mode images with nodule masks drawn by radiologists and benign/malignant labels released by the dataset authors \[37\]. The reference standard underlying those labels is the one described in the dataset publications; no per-image histopathological or cytological detail is available, so no Bethesda-based exclusion or indeterminate subgroup could be formed. TN3K contains no patient identifiers, so the partition is image-level (the official fold-0 training and validation lists and the official test set) and images of the same patient may fall into different partitions; this is stated as a limitation (Section 6). The dataset is de-identified and publicly released under [[dataset licence]]; no ethics approval or consent was required for this secondary analysis. Figure 1 shows the data flow.

### 3.2 Cohort and data sources

**Table 1.** Dataset characteristics (TN3K, official release).

| Item | TN3K |
| --- | --- |
| Source | Zhujiang Hospital, Southern Medical University, China; public release \[35–37\] |
| Collection period | [[ ]] |
| Patients, n | not available (TN3K has no patient identifiers) |
| Images, n | 3,493 (2,303 training / 576 validation / 614 test; official fold-0 lists and official test set) |
| Malignant, n (%) | 1,210 (34.6); by partition 779 (33.8) / 195 (33.9) / 236 (38.4) |
| Benign, n (%) | 2,283 (65.4) |
| Age, median \[IQR\], years | not available |
| Sex, female, n (%) | not available |
| Ultrasound systems (vendor, model, probe) | various devices and views (per the TN3K release); vendor and probe not specified |
| Image size, pixels | 174–826 × 184–1,463 (native, variable); 224 × 224 after nodule-centered crop |
| Reference segmentation | one radiologist-drawn nodule mask per image |
| Inclusion criteria | every image of the official release |
| Exclusion criteria | none |

The official TN3K fold-0 lists define the training (2,303 images) and validation (576) partitions, and the official test set (614) is the internal test partition; the split is image-level because TN3K has no patient identifiers. Every tuning decision (α, β, γ, λ, τ, ensemble weights ω, threshold t\*) was made on the validation partition only. The internal test partition was scored once, after all choices had been frozen.


**Table 2.** Model inputs.

| Modality | Variables | Dimension | Availability on TN3K and missing-value handling |
| --- | --- | --- | --- |
| Ultrasound image | one B-mode image per nodule, nodule-centered square crop (1.5 × the mask bounding box, resized to 224 × 224) | H × W = 224 × 224 | available for every image |
| Structured sonographic descriptors v\_n | area fraction, taller-than-wide ratio, eccentricity, solidity, irregularity, echo ratio (inside vs. surrounding ring), echo mean, echo SD, echo entropy, bright-spot fraction, connected-component count, all computed from the image and the reference mask | d\_v = 11 | available for every image (no empty mask); defined as missing when the mask is empty, then median-imputed on the training partition with an observation indicator per variable (Eq. 10) |
| Laboratory | TSH, FT3, FT4 | d\_l = 3 | not available in TN3K; defined for the clinical deployment of the framework (median imputation on the training partition plus a missing indicator per variable) |
| Clinical | age, sex, ACR TI-RADS category \[2\] (ordinal 1–5), maximum diameter | d\_c = 4 | not available in TN3K; defined for the clinical deployment of the framework |
| Regional | iodine-deficiency index of the patient's region (median urinary iodine, µg/L), regional malignancy prevalence π\_reg | d\_r = 2 | not available in TN3K; defined for the clinical deployment of the framework |
| Cytology / molecular | Bethesda category, [[BRAF V600E status]] | d\_g = [[2]] | not available in TN3K; defined for the clinical deployment of the framework |


### 3.3 Notation

Index n = 1, …, N runs over nodules, p over pixels of one image and k over base learners; bold lower-case letters denote vectors.

| Symbol | Meaning | Range |
| --- | --- | --- |
| I\_n | B-mode image of nodule n | H × W, 8-bit |
| x̃(p) | intensity at pixel p after min–max normalization | \[0, 1\] |
| W\_p | square window of side k = 7 pixels centered at p | — |
| c(p), σ(p) | mean and standard deviation of x̃ over W\_p | \[0, 1\] |
| σ\_g | standard deviation of x̃ over the whole image | \[0, 1\] |
| T(p) | normalized local entropy (texture) | \[0, 1\] |
| K(p) | Michelson local contrast | \[0, 1\] |
| q | image quality coefficient | \[0, 1\] |
| μ(p) | adaptive fuzzy membership of pixel p | \[0, 1\] |
| r(p) | radius of the structuring element at p | integers in \[r\_min, r\_max\] |
| w(p) | reliability weight of the morphological operator at p | \[0, 1\] |
| F | adaptive fuzzy-morphological map of the image | \[0, 1\]^{H×W} |
| l\_n, c\_n, r\_n, g\_n | laboratory, clinical, regional, cytological/molecular vectors (clinical deployment) | ℝ^{d\_l}, ℝ^{d\_c}, ℝ^{d\_r}, ℝ^{d\_g} |
| v\_n | structured sonographic descriptor vector of nodule n (tabular block in the TN3K experiment) | ℝ^{d\_v}, d\_v = 11 |
| Q\_m | data-quality coefficient of modality m | \[0, 1\] |
| h\_img | image embedding from the convolutional backbone | ℝ^{d\_h}, d\_h = 128 |
| z\_n | fused feature vector of nodule n | ℝ^{d\_z} |
| s\_n | sample membership of nodule n in the fuzzy SVM | (0, 1\] |
| p\_k | malignancy probability from learner k | \[0, 1\] |
| ω\_k | ensemble weight of learner k | \[0, 1\], Σω\_k = 1 |
| Δ\_reg | regional prior correction | ℝ |
| t\* | decision threshold | (0, 1) |
| y\_n, ŷ\_n | reference label and predicted label of nodule n (1 = malignant) | {0, 1} |

μ denotes pixel membership only and s sample membership only; α and β appear only in Eq. (4), γ only in Eq. (5), λ only in Eq. (8) and τ only in Eq. (14).

### 3.4 Adaptive texture–contrast fuzzy membership (ATCFM)

Each pixel receives a membership μ(p) in \[0, 1\] that is high inside homogeneous tissue and low at noisy or edge pixels. The image is first min–max normalized:

```latex
\tilde{x}(p) = \frac{I(p) - \min_{q} I(q)}{\max_{q} I(q) - \min_{q} I(q)} \qquad (1)
```

Local statistics are computed over the k × k window W\_p centered at p:

```latex
c(p) = \frac{1}{|W_p|}\sum_{q \in W_p} \tilde{x}(q), \qquad \sigma(p) = \sqrt{\frac{1}{|W_p|}\sum_{q \in W_p} \big(\tilde{x}(q) - c(p)\big)^2} \qquad (2)
```

Three descriptors in \[0, 1\] enter the membership: the texture T(p), defined as the Shannon entropy of the L = 256-bin histogram of W\_p divided by log2 L; the Michelson contrast K(p) of W\_p; and an image quality coefficient q, computed once per image from the noise estimate σ\_noise of the Laplacian method of Immerkær \[26\], with SNR\_0 the median SNR of the training images:

```latex
T(p) = \frac{H(W_p)}{\log_2 L}, \qquad K(p) = \frac{\max_{W_p}\tilde{x} - \min_{W_p}\tilde{x}}{\max_{W_p}\tilde{x} + \min_{W_p}\tilde{x} + \epsilon}, \qquad q = \frac{\mathrm{SNR}}{\mathrm{SNR} + \mathrm{SNR}_0}, \quad \mathrm{SNR} = \frac{\overline{\tilde{x}}}{\sigma_{\text{noise}}} \qquad (3)
```

The adaptive membership is

```latex
\mu(p) = q \cdot \exp\!\left(-\frac{\big(\tilde{x}(p) - c(p)\big)^2}{2\sigma(p)^2 + \epsilon}\right) \cdot \frac{\big(1 + \alpha T(p)\big)\big(1 + \beta K(p)\big)}{(1 + \alpha)(1 + \beta)} \qquad (4)
```

with ε = 10⁻⁶ and α, β ≥ 0 selected on the validation partition over the grid {0, 0.25, 0.5, 1, 2}. Every factor lies in \[0, 1\], so μ(p) ≤ q ≤ 1 for all p. With α = β = 0 and q = 1, Eq. (4) reduces to the local Gaussian membership, so the proposed function strictly generalizes it.

### 3.5 Adaptive fuzzy morphological operations (AFMO)

The structuring element B\_r(p) is a disc whose radius r(p) grows with local heterogeneity, so that complex nodule borders are processed with a larger neighborhood than homogeneous parenchyma:

```latex
r(p) = \operatorname{clip}\!\Big( \operatorname{round}\!\big[ r_0 \big(1 + \gamma\, \sigma(p)/\sigma_g \big) \big],\; r_{\min},\; r_{\max} \Big) \qquad (5)
```

with r\_0 = 2, r\_min = 1, r\_max = 5 pixels and γ selected on the validation partition over {0.5, 1, 2}. A reliability weight limits the operator where the local variance is high:

```latex
w(p) = 1 - \frac{\sigma(p)}{\max_{q} \sigma(q)} \in [0, 1] \qquad (6)
```

The reliability-weighted fuzzy erosion ε\_w and dilation δ\_w of a map f in \[0, 1\]^{H×W} are convex combinations of the classical flat gray-scale operator and the identity:

```latex
\varepsilon_w(f)(p) = w(p) \min_{q \in B_{r(p)}(p)} f(q) + \big(1 - w(p)\big) f(p), \qquad \delta_w(f)(p) = w(p) \max_{q \in B_{r(p)}(p)} f(q) + \big(1 - w(p)\big) f(p) \qquad (7)
```

Both operators map \[0, 1\] to \[0, 1\]; for w(p) = 1 they are the classical erosion and dilation, and for w(p) = 0 they leave f(p) unchanged, so the operator's effect decreases monotonically with local variance. Opening, closing and the final AFMO map are

```latex
O = \delta_w\big(\varepsilon_w(\mu)\big), \qquad C = \varepsilon_w\big(\delta_w(\mu)\big), \qquad F = \lambda\, O + (1 - \lambda)\, C, \quad \lambda \in [0, 1] \qquad (8)
```

Opening suppresses bright speckle smaller than B\_r; closing fills dark gaps inside the nodule. λ was selected on the validation partition over {0, 0.1, …, 1} by the Dice coefficient between F binarized with Otsu's threshold \[27\] and the reference nodule masks released with TN3K \[35, 36\]. The selected λ and the resulting Dice are reported in Section 5.2 as a segmentation result. F is passed to the classifier as a second image channel (Section 3.6), so no information is lost by binarization.

### 3.6 Multimodal feature fusion

The image embedding is produced by a convolutional backbone f\_θ (ResNet-18 \[11\], ImageNet-initialized, first convolution widened to two input channels) applied to the normalized image and the AFMO map stacked as channels:

```latex
\mathbf{h}_{\text{img}} = f_\theta\big([\tilde{x},\, F]\big) \in \mathbb{R}^{d_h} \qquad (9)
```

Each modality m carries a quality coefficient Q\_m. For the image, Q\_img = q from Eq. (3); for a tabular modality with d\_m variables and observation indicators o\_{mj} (1 = measured, 0 = imputed),

```latex
Q_m = \frac{1}{d_m}\sum_{j=1}^{d_m} o_{mj}, \qquad m \in \{\text{lab},\, \text{clin},\, \text{reg}\} \qquad (10)
```

Tabular vectors are median-imputed and z-scored with the training-partition statistics (denoted by a tilde); in clinical deployment, sex is coded 0/1 and TI-RADS is kept ordinal. Because the modality vectors have different dimensions and units, they are fused by concatenation:

```latex
\mathbf{z} = \big[\, Q_{\text{img}}\mathbf{h}_{\text{img}} ;\; Q_{\text{lab}}\tilde{\mathbf{l}} ;\; Q_{\text{clin}}\tilde{\mathbf{c}} ;\; Q_{\text{reg}}\tilde{\mathbf{r}} ;\; Q_{\text{img}}, Q_{\text{lab}}, Q_{\text{clin}}, Q_{\text{reg}} \,\big] \in \mathbb{R}^{d_z}, \quad d_z = d_h + d_l + d_c + d_r + 4 \qquad (11)
```

The four coefficients are appended as features so that tree learners can model interactions between data quality and value. One backbone per seed, trained on the full training partition and early-stopped on validation, produces h\_img for every nodule, so all embeddings share one coordinate system (embeddings from separately trained fold models are not comparable). Training-row embeddings are therefore in-sample; to limit the resulting optimism, every downstream learner is tuned on the validation partition only and the test partition is scored once.

On TN3K the tabular block consists of the descriptor vector v\_n only (Table 2), so Eq. (11) reduces to z = \[Q\_img h\_img ; Q\_struct ṽ ; Q\_img, Q\_struct\] with d\_z = d\_h + d\_v + 2 = 141, where Q\_struct is computed from the descriptors' observation indicators as in Eq. (10), and Δ\_reg = 0 (Section 3.9). The same quality-weighted tabular block is also supplied to the head g\_φ of the DL learner (Section 3.7).

### 3.7 Base learners

Four learners map z to a malignancy probability p\_k in \[0, 1\]. The RF, XGB and FSVM hyperparameters were selected on the validation partition; the DL learner's optimizer settings were fixed a priori and only its stopping epoch was chosen on validation. Grids and selected values are listed in Supplementary Table S2.

| Learner k | Model | Output p\_k | Hyperparameters tuned |
| --- | --- | --- | --- |
| DL | f\_θ trained end-to-end with a two-layer head g\_φ: d\_z → 64 → 1, sigmoid output, binary cross-entropy, AdamW, early stopping on validation AUC | p\_DL = g\_φ(z) | none tuned: learning rate 3 × 10⁻⁴, weight decay 10⁻⁴, dropout 0.3, cosine schedule, at most 30 epochs with patience 6, fixed a priori; stopping epoch by validation AUC |
| RF | random forest \[16\] | out-of-bag-calibrated class fraction | number of trees 500, maximum features, minimum leaf size |
| XGB | gradient-boosted trees \[17\], logistic objective | sigmoid of the boosted score | depth, learning rate, number of rounds, subsample, column sample |
| FSVM | fuzzy SVM \[19\], RBF kernel with width κ | Platt-scaled decision value \[28\], scaling fitted on the validation partition | C, κ |

The fuzzy SVM assigns each training nodule a membership s\_n that lowers the penalty of atypical samples and solves

```latex
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \; \tfrac{1}{2}\|\mathbf{w}\|^2 + C \sum_{n=1}^{N} s_n \xi_n \quad \text{s.t.} \quad y'_n\big(\mathbf{w}^{\top}\phi(\mathbf{z}_n) + b\big) \ge 1 - \xi_n, \;\; \xi_n \ge 0, \;\; y'_n = 2y_n - 1 \qquad (12)
```

with the sample membership defined by the distance to the class centroid z̄\_y \[19\]:

```latex
s_n = 1 - \frac{\|\mathbf{z}_n - \bar{\mathbf{z}}_{y_n}\|}{\max_{m:\, y_m = y_n}\|\mathbf{z}_m - \bar{\mathbf{z}}_{y_n}\| + \delta}, \qquad \delta = 10^{-3} \qquad (13)
```

so that s\_n lies in (0, 1\]. The sample membership s\_n weights nodules inside the SVM loss; the pixel membership μ(p) of Eq. (4) weights pixels inside the image. They are distinct quantities.

### 3.8 Adaptive ensemble

The ensemble probability is a convex combination whose weights depend on each learner's validation log-loss L\_k^val, so that a weak learner cannot dominate:

```latex
p_{\text{ens}} = \sum_{k} \omega_k\, p_k, \qquad \omega_k = \frac{\exp(-L_k^{\text{val}}/\tau)}{\sum_{j} \exp(-L_j^{\text{val}}/\tau)} \qquad (14)
```

The temperature τ > 0 was selected on the validation partition over {0.05, 0.1, 0.2, 0.5, 1}; τ → ∞ gives equal weights and τ → 0 selects the single best learner. Weights were frozen before the test partition was scored.

### 3.9 Regional epidemiological prior

Regional prevalence enters as a Bayesian prior-shift correction \[29\]:

```latex
\Delta_{\text{reg}} = \operatorname{logit}\pi_{\text{reg}} - \operatorname{logit}\pi_{\text{train}} \qquad (15)
```

where π\_train is the malignant fraction of the training partition and π\_reg the malignancy prevalence among biopsied nodules in the deployment region, taken from [[regional cancer registry or institutional data, year]]. When the deployment region equals the training region, Δ\_reg = 0. The iodine-deficiency index is a feature inside r\_n (Table 2), not a multiplier; both regional inputs are removed in the ablation of Section 3.12. In the present experiment all images come from one source and no regional variables exist (Table 2), so r\_n is absent and Δ\_reg = 0 throughout; Eq. (15) is defined but not exercised here.

### 3.10 Decision rule

The final probability and predicted label are

```latex
p_{\text{final}} = \sigma\!\Big(\operatorname{logit} p_{\text{ens}} + \Delta_{\text{reg}}\Big), \qquad \hat{y} = \mathbb{1}\big[p_{\text{final}} \ge t^{*}\big] \qquad (16)
```

The threshold t\* was fixed on the validation partition before testing by the pre-specified rule sensitivity ≥ 0.95 with maximal specificity; the Youden-optimal threshold is reported alongside. Equations (1)–(16) define TAMES AI completely; every symbol is listed in Section 3.3.

### 3.11 Explainability

Feature attributions were computed in probability space with SHAP \[8\]: TreeSHAP for RF and XGB, KernelSHAP with a 100-nodule training background for FSVM and the DL head. Because p\_ens in Eq. (14) is linear in the learner outputs, the ensemble attribution of feature j is the weighted sum of the learner attributions, φ\_j^ens = Σ\_k ω\_k φ\_j^k, with no further approximation. Image evidence is visualized with Grad-CAM \[9\] on the last convolutional block of f\_θ.

Explanation quality was measured rather than assumed: Grad-CAM faithfulness by deletion and insertion curves \[30\] on the internal test partition; SHAP stability by the rank of the ten most important features across 2,000 bootstrap resamples. A clinical plausibility rating by radiologists was not performed in this study.

### 3.12 Evaluation and statistical analysis

The statistical analysis plan was fixed before the test partition was scored ([[date; repository or OSF link]]). All confidence intervals are 95 % intervals from 2,000 image-level bootstrap resamples (TN3K has no patient identifiers).

| Task | Metrics | Comparison test |
| --- | --- | --- |
| Benign–malignant classification | AUC; sensitivity, specificity, PPV, NPV, accuracy, F1 and MCC at t\* | AUC: DeLong test \[31\] (difference, CI, z, p); sensitivity and specificity at t\*: exact McNemar test \[32\] with discordant counts; Holm correction \[33\] over all baseline comparisons |
| Calibration | Brier score, calibration slope and intercept, calibration curve (10 quantile bins) | — |
| Segmentation (AFMO only) | Dice of Otsu-binarized F against the reference masks | paired Wilcoxon signed-rank test against Otsu on the raw image |
| Explanation quality | Grad-CAM deletion and insertion AUC against a random-saliency control; SHAP top-10 rank stability | — |

Baselines were trained on the same partitions with the same tuning budget: image-only CNN (x̃ channel only, no tabular input); descriptor-only XGBoost; CNN + XGBoost late fusion (average of the two probabilities); and each of the four base learners alone. A TI-RADS-alone baseline is impossible on TN3K, which has no TI-RADS category.

Ablation removed one component at a time and re-tuned the remainder on the validation partition: (a) AFMO off, F replaced by μ; (b) ATCFM off, μ replaced by the Gaussian term of Eq. (4) with α = β = 0 and q = 1; (c) quality coefficients off, Q\_m = 1; (d) regional inputs off, r\_n removed and Δ\_reg = 0, which is not applicable on TN3K because no regional inputs exist and Δ\_reg = 0 already, so (d) coincides with the full model; (e) equal ensemble weights, τ → ∞; (f) each learner removed in turn. Differences are reported with bootstrap intervals. Subgroup performance is reported by nodule size only, the sole subgroup derivable from TN3K (no Bethesda, sex or scanner metadata exist); size is measured as the mask area fraction of the image, because no pixel spacing is available, and split at the training-partition median (0.082).

The minimum development sample was computed with the criteria of Riley et al. \[34\] for a binary outcome with d\_z = 141 candidate predictors, an anticipated AUC of [[ ]] and a prevalence of 0.338 (training and validation partitions); the requirement of [[n]] nodules with [[e]] events is [[met / not met]] by the development cohort.

### 3.13 Implementation

| Item | Value |
| --- | --- |
| Framework | PyTorch 2.14.1 for f\_θ and g\_φ; scikit-learn 1.9.1 for RF and FSVM; xgboost 3.4.1; shap 0.53.0 |
| Hardware | Apple M5 Pro (MPS backend), 24 GB unified memory; backbone training 3.5–6.2 min per seed (22–30 epochs; from output timestamps) |
| Seeds | five runs (seeds 0–4); metrics computed on predictions averaged over the five seeds, with bootstrap CIs; seed-to-seed SD of the test AUC in Supplementary Table S4 |
| Image pipeline | nodule-centered crop to 224 × 224; window k = 7; L = 256; ε = 10⁻⁶; augmentation: horizontal flip, ±10° rotation, brightness ±10 % |
| Selected hyperparameters | α, β, γ, λ, τ, t\* and all learner settings in Supplementary Table S2, with the validation value that selected them |
| Code and weights | [[public repository URL, commit hash]]; trained weights [[DOI]] |

## 4. System Architecture

TAMES AI is organized as eight layers (Figure 2). Four input modalities enter the data-collection layer (in the TN3K experiment, the image and the descriptor vector only). The image passes through the adaptive fuzzy layer (Sections 3.4–3.5) and the convolutional backbone, while laboratory, clinical and regional data pass through preparation and quality scoring; the fusion layer builds z (Eq. 11); the hybrid AI layer runs the four learners; the adaptive ensemble, regional prior and decision rule produce the final probability (Eqs. 14–16); the explainable-AI layer attaches SHAP and Grad-CAM evidence; and the clinical decision-support layer returns the label, probability, explanation and a TI-RADS-aligned report to the clinician. Inference takes [[x]] s per nodule on [[GPU]].

&#91;embedded content: TAMES AI architecture · 8 layers from four inputs to one explained probability\]

**Figure 2.** Architecture of TAMES AI. The highlighted layer is the contribution of this work; equation numbers refer to Section 3.

In the TN3K experiment the layers are instantiated as follows. The adaptive fuzzy layer turns each 224 × 224 crop into the two-channel tensor \[x̃, F\] (Eqs. 1–8; k = 7, r\_0 = 2, r\_min = 1, r\_max = 5, α = 2, β = 0.5, γ = 2, λ = 0 after validation selection). The backbone f\_θ is an ImageNet-initialized ResNet-18 whose first convolution takes two channels; its 512-dimensional pooled output passes through a linear–ReLU–dropout block to the embedding h\_img ∈ ℝ^128 (Eq. 9). The structured branch is the eleven-descriptor vector v\_n with its quality coefficient Q\_struct (Eq. 10), and the fusion layer forms z ∈ ℝ^141 (Eq. 11). The DL learner's head g\_φ is a two-layer network 141 → 64 → 1 trained jointly with f\_θ; the RF (500 trees), XGBoost and fuzzy SVM learners are fitted on the frozen z of the training partition and tuned on validation. The ensemble layer applies Eq. (14) with τ chosen on validation, Δ\_reg = 0 in Eq. (16), and the threshold t\* fixed on validation; the explainable-AI layer computes TreeSHAP/KernelSHAP attributions on z and Grad-CAM on the last residual block. The clinical decision-support layer (report generation) is not evaluated in this study.

## 5. Results

### 5.1 Cohort

TN3K contains 3,493 images, of which 1,210 (34.6 %) are labeled malignant; no image was excluded. The official fold-0 training, validation and test partitions hold 2,303 (779 malignant, 33.8 %), 576 (195, 33.9 %) and 614 (236, 38.4 %) images (Table 1). Every image has a reference mask, so all eleven descriptors were observed for every image and Q\_struct = 1 throughout; the image-quality coefficient q (Eq. 3) had a mean of 0.494 on the training and 0.531 on the test partition. No laboratory, clinical, regional, cytological or molecular variables exist in TN3K (Table 2).


### 5.2 Segmentation

Grid search on 150 validation images selected α = 2, β = 0.5, γ = 2 and λ = 0 (bright polarity; λ = 0 means F equals the fuzzy closing C), with a validation Dice of 0.245 against 0.224 for the better raw-Otsu polarity. With these parameters the Otsu-binarized AFMO map reached a Dice coefficient of 0.201 \[0.188, 0.214\] against the reference masks on the internal test partition, compared with 0.120 \[0.110, 0.131\] for Otsu thresholding of the raw image (paired Wilcoxon signed-rank test, n = 614, p < 0.001). F is a homogeneity map binarized by a whole-image threshold without post-processing, so this is a segmentation-stage result, not a segmentation benchmark. Supplementary Figure S1 shows representative maps.

### 5.3 Classification

**Table 3.** Internal test partition (614 nodule images): discrimination and metrics at each model's pre-specified threshold t\* (sensitivity ≥ 0.95 on validation; TAMES AI t\* = 0.197, Youden threshold 0.404); 95 % bootstrap CIs in brackets. Metrics are computed on test probabilities averaged over the five seeds.

| Model | AUC | Sensitivity (%) | Specificity (%) | Accuracy (%) | F1 (%) | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Image-only CNN | 0.750 \[0.712, 0.788\] | 90.7 \[86.9, 94.0\] | 33.1 \[28.3, 37.9\] | 55.2 \[51.3, 59.3\] | 60.9 \[56.3, 65.0\] | 0.271 \[0.204, 0.334\] |
| Descriptor-only XGBoost | 0.684 \[0.644, 0.728\] | 78.8 \[73.7, 83.8\] | 47.1 \[42.2, 52.1\] | 59.3 \[55.4, 63.2\] | 59.8 \[55.0, 64.2\] | 0.261 \[0.189, 0.332\] |
| CNN + XGBoost late fusion | 0.751 \[0.712, 0.789\] | 86.9 \[82.2, 90.9\] | 50.0 \[44.9, 55.2\] | 64.2 \[60.3, 68.1\] | 65.1 \[60.7, 69.5\] | 0.374 \[0.307, 0.441\] |
| DL learner alone | 0.754 \[0.714, 0.793\] | 86.0 \[81.3, 90.4\] | 47.6 \[42.7, 52.8\] | 62.4 \[58.6, 66.4\] | 63.7 \[59.1, 68.2\] | 0.344 \[0.277, 0.413\] |
| RF learner alone | 0.738 \[0.698, 0.778\] | 84.3 \[79.5, 88.9\] | 41.5 \[36.8, 46.5\] | 58.0 \[53.9, 62.1\] | 60.7 \[55.8, 65.2\] | 0.271 \[0.197, 0.341\] |
| XGB learner alone | 0.746 \[0.706, 0.786\] | 87.7 \[83.3, 91.7\] | 47.4 \[42.4, 52.6\] | 62.9 \[59.0, 66.9\] | 64.5 \[59.8, 68.7\] | 0.360 \[0.294, 0.428\] |
| FSVM learner alone | 0.737 \[0.696, 0.779\] | 83.9 \[79.2, 88.4\] | 44.4 \[39.5, 49.5\] | 59.6 \[55.9, 63.7\] | 61.5 \[56.8, 66.0\] | 0.292 \[0.222, 0.361\] |
| TAMES AI | 0.750 \[0.710, 0.790\] | 82.2 \[77.0, 87.4\] | 54.2 \[49.3, 59.5\] | 65.0 \[61.2, 69.1\] | 64.3 \[59.6, 68.8\] | 0.361 \[0.293, 0.434\] |

A TI-RADS-alone baseline could not be run because TN3K provides no TI-RADS category.

The AUC difference between TAMES AI and the CNN + XGBoost late-fusion baseline was −0.001 \[−0.026, 0.023\] (DeLong z = −0.09, p = 0.927, Holm-corrected p = 1.000). At t\*, sensitivity differed by −4.7 percentage points and specificity by +4.2 points; exact McNemar tests on discordant pairs (10 vs 21 for sensitivity, 49 vs 33 for specificity) gave p = 0.071 and p = 0.097. AUC differences of TAMES AI against the other baselines were: the image-only CNN 0.000 (Holm p = 1.000); the descriptor-only XGBoost +0.066 (Holm p = 0.012); the DL learner alone −0.004 (Holm p = 0.916); the RF learner alone +0.012 (Holm p = 0.012); the XGB learner alone +0.004 (Holm p = 0.856); the FSVM learner alone +0.012 (Holm p = 0.323). Figure 3 shows the ROC curves of all models. The threshold t\* fixed on validation for sensitivity ≥ 0.95 gave a test sensitivity of 82.2 %, which indicates a validation-to-test shift: the official test set has a higher malignant fraction (38.4 % vs 33.9 %) and a higher mean SNR (76.8 vs 63.0) than the validation partition, and the backbone's validation AUC (0.826–0.842 across seeds) exceeded its test AUC (0.714–0.750). Seed-to-seed standard deviation of the TAMES AI test AUC was 0.012 (per-seed mean 0.724) (Supplementary Table S4).

### 5.4 External validation

External validation was not performed: no external cohort was available for this analysis (see README §7 on the planned DDTI loader).

### 5.5 Calibration

On the internal test partition the Brier score was 0.199, the calibration slope 0.77 \[0.62, 0.94\] and the intercept 0.20 \[−0.01, 0.40\] (slope and intercept CIs from 2000 bootstrap resamples of the pooled test predictions). Calibration curves are shown in Figure 4.

### 5.6 Ablation

**Table 4.** Change in internal-test AUC when one component is removed and the rest re-tuned; 95 % bootstrap CIs in brackets.

| Removed component | AUC | ΔAUC vs full model |
| --- | --- | --- |
| None (full TAMES AI) | 0.750 \[0.710, 0.790\] | — |
| (a) AFMO (F replaced by μ) | 0.744 | −0.005 \[−0.022, 0.010\], DeLong p = 0.484 |
| (b) ATCFM (plain Gaussian membership) | 0.740 | −0.010 \[−0.030, 0.009\], DeLong p = 0.317 |
| (c) Quality coefficients (Q\_m = 1) | 0.748 | −0.002 \[−0.015, 0.011\], DeLong p = 0.768 |
| (d) Regional inputs (r\_n and Δ\_reg) | not applicable on TN3K (no regional inputs, Δ\_reg = 0 already; identical to the full model) | 0 |
| (e) Adaptive weights (equal ω\_k) | 0.748 | −0.002 \[−0.004, 0.001\], DeLong p = 0.268 |
| (f1) DL learner | 0.746 | −0.004 \[−0.008, −0.000\], DeLong p = 0.042 |
| (f2) RF learner | 0.750 | +0.001 \[−0.001, 0.003\], DeLong p = 0.508 |
| (f3) XGB learner | 0.750 | +0.000 \[−0.004, 0.004\], DeLong p = 0.928 |
| (f4) FSVM learner | 0.749 | −0.001 \[−0.005, 0.003\], DeLong p = 0.622 |

The largest loss followed removal of ATCFM (b) (ΔAUC −0.010 \[−0.030, 0.009\]); removal of AFMO (a), ATCFM (b), the quality coefficients (c), the adaptive weights (e), the RF learner (f2), the XGB learner (f3), the FSVM learner (f4) did not change AUC beyond the confidence interval of the difference. The learned ensemble weights, averaged over the five seeds (± SD), were ω\_DL = 0.202 ± 0.062, ω\_RF = 0.091 ± 0.099, ω\_XGB = 0.338 ± 0.036, ω\_FSVM = 0.369 ± 0.109 at τ = 0.05 in all five seeds.

### 5.7 Subgroups

**Table 5.** Internal-test performance by nodule size (mask area fraction, split at the training-partition median of 0.082); 95 % bootstrap CIs in brackets.

| Subgroup | n | AUC | Sensitivity (%) | Specificity (%) |
| --- | --- | --- | --- | --- |
| Smaller nodules (area fraction < 0.082) | 339 (171 malignant) | 0.696 \[0.639, 0.747\] | 85.4 \[79.7, 90.4\] | 35.7 \[28.5, 42.8\] |
| Larger nodules (area fraction ≥ 0.082) | 275 (65 malignant) | 0.734 \[0.656, 0.803\] | 73.8 \[63.1, 83.9\] | 69.0 \[62.7, 75.1\] |

Subgroups by Bethesda category, ultrasound system and sex could not be formed because TN3K carries none of this metadata.


### 5.8 Explanations

The ten features with the highest mean absolute ensemble SHAP value were, in order, the image embedding h\_img (summed over its 128 dimensions), solidity, taller-than-wide ratio, eccentricity, margin irregularity, area fraction, component count, echo entropy, echo SD, echo mean (ensemble attribution of 200 test images, Supplementary Table S5); 9 of the ten stayed in the top ten in at least 95 % of 2000 bootstrap resamples, the least stable being echo mean (77 %) (Figure 5a). Grad-CAM deletion and insertion AUCs on 32 test images were 0.321 and 0.475, against 0.410 and 0.411 for a random-saliency control (lower deletion and higher insertion AUC indicate a faithful map) (Figure 5b).

## 6. Discussion

On TN3K, TAMES AI did not improve discrimination over the imaging-only network, the descriptor-only model and the CNN + XGBoost late fusion (Table 3), and the ablation locates the largest change in ATCFM (b) (Table 4). Three design choices were expected to drive the result; on TN3K their measured contributions were small (Table 4), and the image-only CNN, the late fusion and the full model reached the same test AUC within 0.002 (Table 3), so the descriptor branch and the fuzzy channels added little discriminative information beyond the image on this dataset. The fuzzy membership of Eq. (4) and the reliability-weighted morphology of Eqs. (6)–(8) give the backbone a second channel in which speckle is suppressed and nodule borders preserved, which the segmentation result in Section 5.2 supports, although removing F cost only ΔAUC −0.005 in classification. Quality-weighted concatenation of Eq. (11) lets the learners discount a modality when it is noisy or imputed instead of trusting all inputs equally; on TN3K only Q\_img varied, because every descriptor was observed. The validation-loss-dependent weights of Eq. (14) keep the ensemble from being dominated by a learner that fits the training data best but generalizes worst.

The regional prior of Eq. (15) is a transparent way to move the model between populations with different prevalence without retraining; it was not exercised here (Δ\_reg = 0, single-source data) and its effect remains to be measured on an external cohort. The same holds for the laboratory, clinical and regional branches of Eq. (11): they are defined and implemented, but no TN3K variable feeds them, so the present results concern the image-plus-descriptor instantiation only.

Compared with imaging-only AI in pooled analyses \[4–6\], the AUC reported here (0.750) is below the pooled range of those image-only classifiers (above 0.90); no external validation was performed, so the transfer gap documented in two-center validations of imaging-only systems \[15\] could not be assessed. The explanation audit adds information that most published models omit: Grad-CAM maps were more faithful than a random-saliency control, and the SHAP ranking was partly stable under resampling (Section 5.8).


**Limitations.** The study uses a single public dataset from one institution, with the reference labels released by its authors and no per-image histopathological or cytological detail. Because TN3K has no patient identifiers, the split is image-level, so images of the same patient may fall into different partitions and the test estimates may be optimistic. The structured descriptors are computed from a radiologist's nodule delineation, so the descriptor branch depends on a manual segmentation at inference time; the fully automatic alternative (descriptors from the Otsu-binarized F) was not evaluated. The laboratory, clinical and regional branches and the regional prior of Eq. (15) are defined but untested, and no cytological or molecular data were available. No external, prospective or reader-study evaluation was performed, so neither transfer to other scanners and populations nor the effect of the model on clinical decisions is known. Membership functions and morphological parameters were tuned by grid search on one validation partition, and training-row embeddings are in-sample (Section 3.6); nested cross-validation would give less optimistic selection.


## 7. Conclusion

Representing uncertainty explicitly at the pixel, modality and sample levels, and fusing ultrasound with structured sonographic descriptors, produced a thyroid-nodule classifier that did not outperform single-modality and late-fusion baselines on the public TN3K dataset. The laboratory, clinical and regional branches of the framework and the regional prior remain to be validated on Central Asian clinical cohorts. TAMES AI is a research-stage decision-support tool, not an autonomous diagnostic device. External validation, prospective multicenter evaluation, a reader study measuring the change in biopsy and surgery decisions, and privacy-preserving training across institutions are the next steps.

## Declarations

**Data availability.** The TN3K images, masks and split lists are publicly available from the dataset authors \[35, 36\] ([[URL]]); the benign/malignant labels are those of the ACL release \[37\] ([[URL]]). No new patient data were collected.

**Code availability.** Training and evaluation code, the statistical analysis plan and the trained weights are available at [[repository URL, commit hash, DOI]].

**Ethics approval and consent.** Not required: this study is a secondary analysis of a public, de-identified dataset.

**Funding.** [[none / grant number]].

**Competing interests.** The authors declare no competing interests.

**Author contributions.** [[N.N.Z.: conceptualization, methodology, software, analysis, writing – original draft. M.A.M.: supervision, methodology, writing – review and editing. Both authors approved the final manuscript.]]

## References

1. Haugen BR, Alexander EK, Bible KC, et al. 2015 American Thyroid Association management guidelines for adult patients with thyroid nodules and differentiated thyroid cancer. Thyroid. 2016;26(1):1–133. https://doi.org/10.1089/thy.2015.0020
2. Tessler FN, Middleton WD, Grant EG, et al. ACR Thyroid Imaging, Reporting and Data System (TI-RADS): white paper of the ACR TI-RADS Committee. J Am Coll Radiol. 2017;14(5):587–595. https://doi.org/10.1016/j.jacr.2017.01.046
3. Cibas ES, Ali SZ. The 2017 Bethesda System for Reporting Thyroid Cytopathology. Thyroid. 2017;27(11):1341–1346. https://doi.org/10.1089/thy.2017.0500
4. Chen M, et al. A comparison of artificial intelligence versus radiologists in the diagnosis of thyroid nodules using ultrasonography: a systematic review and meta-analysis. Eur Arch Otorhinolaryngol. 2022. https://doi.org/10.1007/s00405-022-07436-1
5. Huang J, et al. Artificial intelligence in thyroid ultrasound imaging: a systematic review and meta-analysis. Eur Radiol. 2023;33. [[verify volume, pages, DOI]]
6. Li X, et al. Artificial intelligence-assisted diagnosis of thyroid nodules based on ultrasound imaging: a systematic review. Front Oncol. 2020;10. [[verify article number, DOI]]
7. Zadeh LA. Fuzzy sets. Inf Control. 1965;8(3):338–353. https://doi.org/10.1016/S0019-9958(65)90241-X
8. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. Adv Neural Inf Process Syst. 2017;30:4765–4774.
9. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: visual explanations from deep networks via gradient-based localization. Proc IEEE Int Conf Comput Vis (ICCV). 2017:618–626. https://doi.org/10.1109/ICCV.2017.74
10. LeCun Y, Bengio Y, Hinton G. Deep learning. Nature. 2015;521:436–444. https://doi.org/10.1038/nature14539
11. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. Proc IEEE Conf Comput Vis Pattern Recognit (CVPR). 2016:770–778. https://doi.org/10.1109/CVPR.2016.90
12. Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. In: MICCAI 2015. Springer; 2015:234–241. https://doi.org/10.1007/978-3-319-24574-4\_28
13. Chen Y, Gao Z, He Y, et al. An artificial intelligence model based on ACR TI-RADS characteristics for US diagnosis of thyroid nodules. Radiology. 2022;303(3):613–619. https://doi.org/10.1148/radiol.211455
14. Ganne NS, Balakrishna S. EvoThy-Net: an evolutionary encoder–decoder network for thyroid nodule segmentation in ultrasound imaging. Sci Rep. 2026;16:4566. [[verify DOI]]
15. &#91;\[authors]]. External validation of a deep learning-based artificial intelligence system for ultrasound diagnosis of thyroid nodules: a two-center retrospective study. J Clin Ultrasound. 2026. [[verify authors, volume, pages, DOI]]
16. Breiman L. Random forests. Mach Learn. 2001;45:5–32. https://doi.org/10.1023/A:1010933404324
17. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. Proc 22nd ACM SIGKDD Int Conf Knowl Discov Data Min. 2016:785–794. https://doi.org/10.1145/2939672.2939785
18. &#91;\[authors]]. Diagnostic performance of ultrasound characteristics-based artificial intelligence models for thyroid nodules: a systematic review and meta-analysis. Front Oncol. 2025. [[verify authors, article number, DOI]]
19. Lin C-F, Wang S-D. Fuzzy support vector machines. IEEE Trans Neural Netw. 2002;13(2):464–471. https://doi.org/10.1109/72.991432
20. Saini M, Adl Parvar T, Velarde M, Larson NB, Fatemi M, Alizad A, et al. Deep learning-based classification of thyroid nodules using uncertainty-aware multi-modal ultrasound imaging. Sci Rep. 2026. [[verify volume, article number, DOI]]
21. &#91;\[authors]]. Multi-modal feature integration for thyroid nodule prediction: combining clinical data with ultrasound-based deep features. J Radiat Res Appl Sci. 2025;18(1):101217. https://doi.org/10.1016/j.jrras.2024.101217 [[verify authors]]
22. Tessler R, [[co-authors]]. Artificial intelligence and machine learning in thyroid ultrasound: current applications and future perspectives. J Clin Med. 2022;11. [[verify authors, article number, DOI]]
23. U.S. Food and Drug Administration. Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan. FDA Center for Devices and Radiological Health; 2021.
24. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
25. Mongan J, Moy L, Kahn CE Jr. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): a guide for authors and reviewers. Radiol Artif Intell. 2020;2(2):e200029. https://doi.org/10.1148/ryai.2020200029
26. Immerkær J. Fast noise variance estimation. Comput Vis Image Underst. 1996;64(2):300–302. https://doi.org/10.1006/cviu.1996.0060
27. Otsu N. A threshold selection method from gray-level histograms. IEEE Trans Syst Man Cybern. 1979;9(1):62–66. https://doi.org/10.1109/TSMC.1979.4310076
28. Platt JC. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In: Smola AJ, Bartlett P, Schölkopf B, Schuurmans D, eds. Advances in Large Margin Classifiers. MIT Press; 2000:61–74.
29. Saerens M, Latinne P, Decaestecker C. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. Neural Comput. 2002;14(1):21–41. https://doi.org/10.1162/089976602753284446
30. Petsiuk V, Das A, Saenko K. RISE: randomized input sampling for explanation of black-box models. Proc Br Mach Vis Conf (BMVC). 2018.
31. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics. 1988;44(3):837–845. https://doi.org/10.2307/2531595
32. McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika. 1947;12(2):153–157. https://doi.org/10.1007/BF02295996
33. Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat. 1979;6(2):65–70.
34. Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a clinical prediction model. BMJ. 2020;368:m441. https://doi.org/10.1136/bmj.m441
35. Gong H, Chen G, Wang R, et al. Multi-task learning for thyroid nodule segmentation with thyroid region prior. Proc IEEE Int Symp Biomed Imaging (ISBI). 2021. https://doi.org/10.1109/ISBI48211.2021.9434087 [[verify authors, pages]]
36. Gong H, Chen J, Chen G, et al. Thyroid region prior guided attention for ultrasound segmentation of thyroid nodules. Comput Biol Med. 2023;155:106389. [[verify authors, volume, DOI]]
37. Gong H, et al. Less is more: adaptive curriculum learning for thyroid nodule diagnosis. In: MICCAI 2022. Springer; 2022. [[verify authors, pages, DOI]]
