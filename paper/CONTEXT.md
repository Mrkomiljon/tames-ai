# Context for whoever edits the paper (human or Claude Code)

This file carries everything from the planning conversation that the manuscript editor
needs and that is not visible in the repository itself.

## 1. What happened to the first submission

The article (Zufarova & Mirzayev, "TAMES AI … for Early Detection of Thyroid Cancer") was
rejected. The reviewers' points, verbatim in substance:

1. *Self-referential reviewer notes inside the main text* (Sections 3.1, 3.2, 3.9, 3.10, 5,
   7, 9): sentences such as "important methodological information … is not specified in the
   source manuscript; these items should be added before journal submission" and "these
   results … should not be interpreted as definitive evidence". Verdict: an unfinished
   internal draft, not a research paper.
2. *Complete absence of dataset and cohort details*: no patient or nodule counts, class
   distribution, imaging sources, scanner specifications, inclusion/exclusion criteria.
3. *No experimental rigor*: no train/validation/test partition, no patient-level
   independence, no hyperparameter protocol, no baseline architectures.
4. *Corrupted mathematics*: syntax errors in 3.3–3.5 and 3.7 ("exp +", "μ( ) ="), two
   subsections numbered 3.5, incomplete summation bounds; the central model
   Y = Decision(Σ_k λ_k M_k(Σ_i Q_i μ_i(x_i) T_i K_i E_i I_i A_i G_i F_i)) concatenates
   undefined symbols; Z and F_AFM = λO + (1−λ)C leave O, C and normalization undefined.

The reported numbers (accuracy 97.2 %, AUC 0.981 vs 94.1 % / 0.964 for a CNN + XGBoost
baseline) had no experiment behind them that could be documented. They must not reappear
unless an experiment under the protocol of Section 3 reproduces them.

## 2. Decisions already taken (do not reopen)

* Equations (1)–(16) in `manuscript_v2.md` are the corrected model. Every symbol is in the
  notation table (3.3). Each equation is implemented in `tames/` (README has the map). The
  equations are frozen: fix the text around them, never the equations.
* Fixes made relative to the original draft, and why (needed for the response letter):
  ATCFM normalized by (1+α)(1+β) so μ ∈ [0,1]; one definition of μ instead of two;
  w(p) = 1 − σ/max σ used as a convex weight so that high variance *reduces* the operator
  (the original w = 1 − σ could go negative and amplified erosion); O and C defined as
  fuzzy opening/closing; fusion by concatenation (a weighted sum of vectors of different
  dimension was undefined); the FSVM sample membership s_n is a different object from the
  pixel membership μ(p); ensemble weights from validation log-loss; regional prevalence as a
  Bayesian prior shift (Eq. 15) rather than a multiplier; the undefined Y = Decision(·)
  withdrawn.
* Embeddings: one CNN per seed trained on the full training partition produces h_img for all
  rows (training rows in-sample). The "out-of-fold embeddings" idea was dropped because fold
  models have incompatible embedding spaces. Downstream learners are tuned on validation only.

## 3. The experiment that actually ran (this is what the paper must describe)

| Item | Reality in this repository |
|---|---|
| Data | TN3K (Gong et al. ISBI 2021 / CBM 2022; labels from Gong et al. MICCAI 2022): 3,493 B-mode images with radiologist nodule masks, benign/malignant labels, "various devices and views", Zhujiang Hospital, Southern Medical University, China |
| Split | official fold-0 lists: train 2,303 / val 576 / official test 614; **image-level** (TN3K has no patient IDs) |
| Reference standard | labels as released by the dataset authors; no histopathology/cytology detail available |
| Modalities used | ultrasound image (x̃, F channels) + **structured sonographic descriptors computed from the image and the dataset mask** (`tames/features/descriptors.py`: area fraction, taller-than-wide, eccentricity, solidity, irregularity, echo ratio/mean/std/entropy, bright-spot fraction, component count) |
| Modalities NOT available | laboratory (TSH, FT3, FT4), clinical (age, sex, TI-RADS, diameter), regional (iodine index, prevalence), cytology/molecular. Δ_reg = 0 throughout. |
| Baselines run | image-only CNN, structured-only XGBoost, CNN + XGBoost late fusion (average), each learner alone. TI-RADS-alone impossible (no TI-RADS in TN3K). |
| External validation | not performed (DDTI loader not implemented) |
| Subgroups | only nodule size is derivable (from the mask); no Bethesda, sex or scanner metadata |
| Seeds | 5 (0–4); pooled predictions for CIs; seed-to-seed SD reported |
| Threshold | t* = sensitivity ≥ 0.95 with maximal specificity on validation; Youden also reported |

## 4. What must be reconciled in `manuscript_v2.md`

The manuscript still describes the *full framework* (lab + clinical + regional inputs,
patient-level 70/15/15 split, external cohort). Keep the framework as the **proposed
method**, but every claim about what was *done* must match §3. Concretely:

1. **Title/Abstract.** Methods sentence: data = TN3K public dataset, structured branch =
   mask-derived sonographic descriptors, no laboratory/regional inputs were available;
   Results: internal test only; Conclusions: framework proposed, image + descriptor
   instantiation evaluated; laboratory, clinical and regional branches and the regional
   prior remain to be validated on clinical cohorts.
2. **3.1.** Replace the histopathology/Bethesda reference-standard text with the TN3K label
   provenance; say the split is image-level and why; remove ethics/consent sentences
   (public de-identified data; cite the dataset licence instead); keep TRIPOD+AI/CLAIM.
3. **3.2.** Table 1: one cohort column (TN3K), delete the external column; Table 2: rows
   for image and *structured descriptors*; lab/clinical/regional rows become "not
   available in TN3K; defined for the clinical deployment of the framework".
4. **3.6 / 3.9.** Eq. (11) general form stays; add one sentence: "on TN3K the tabular block
   consists of the descriptor vector only, so z = [Q_img h_img ; Q_struct s̃ ; Q_img, Q_struct]
   and Δ_reg = 0".
5. **3.12.** Baselines list without TI-RADS-alone; ablation (d) stated as not applicable on
   TN3K; subgroup list reduced to nodule size; sample-size paragraph with TN3K prevalence.
6. **Section 5.** 5.4 external validation and 5.7 non-size subgroups: state "not performed"
   in one sentence each or delete; fill the rest from the run outputs only.
7. **Discussion / Limitations.** Add: single public dataset, image-level split, descriptors
   depend on a radiologist delineation, laboratory and regional branches untested, no
   external or prospective validation.
8. **Response letter** (`response_to_reviewers.md`): rows 2 and 3 must describe the TN3K
   design, not the clinical-cohort design.

Rules: numbers only from `runs/` and `data/processed/` (FILL_RESULTS.md); anything not
available stays `[[...]]`; no sentence may claim something the repository did not do.
