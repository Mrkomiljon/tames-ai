# Figure 2 v3 — notes

## Deliverables
- Script (rewritten in place): `/home/claude/tames/scripts/plots/fig2_architecture.py` — md5 `18457b5d4f814745626935d5d2f5f6be` (552 lines). `figstyle_diagram.py` unchanged (md5 `acae06b5c86d74a6bee5c77ae10896eb`). Nothing else under `/home/claude/tames` was modified. The script sets `sys.dont_write_bytecode` before importing `tames`, so it writes no `__pycache__` into the package.
- Renders: `/tmp/claude-0/-home-claude/56ea871a-75b5-5134-8ca9-904d6cf6bcac/scratchpad/fig2_v3/fig2_architecture.pdf` (vector, 170.00 × 166.15 mm, TrueType fonts embedded) and `/tmp/claude-0/-home-claude/56ea871a-75b5-5134-8ca9-904d6cf6bcac/scratchpad/fig2_v3/fig2_architecture.png` (4015 × 3921 px at 600 dpi = 169.97 × 165.99 mm).
- Render command: `cd /home/claude/tames && python scripts/plots/fig2_architecture.py --out <dir>` (add `--no-maps` to draw panel (b) without the example maps).

## Technical facts and caveats
- **Width.** `bbox_inches="tight"` includes the full axes and then adds `pad_inches` = 0.508 mm on each side. The canvas is therefore 170 − 1.016 mm wide, so the saved page is exactly 170 mm; every drawn element lies at x ≤ 168 mm.
- **Font.** `setup_font()` picks **Liberation Sans** on this machine, not DejaVu Sans. Liberation Sans has the same metrics as Arial and Helvetica, so the measured fit is what the Mac will produce. As an extra check I ran the script with DejaVu Sans forced (about 8 % wider): the layout check is also clean.
- **Layout check.** It runs on every render (`check_layout()`). It measures every text with the renderer and reports: text outside its own box (0.35 mm side padding); text crossing any other box; text–text overlaps; base size below 5.5 pt; anything beyond x = 168 mm. Final result: *clean*, with 168 texts and 46 boxes. The tightest horizontal margin is 0.83 mm ("BN, ReLU" in the stem).
- **Sizes used (pt).** 8 (panel letters), 7 (titles), 6 (lines), 5.5 (shapes and edge labels). There is no base text below 5.5 pt.
  - Residual: mathtext sets sub- and superscripts at 70 % of the base size (about 3.9–4.9 pt). Elsevier's rule of thumb asks for sub/superscripts ≥ 6 pt, which would need a base of about 8.6 pt. I avoided double scripts (for example ω_k ∝ exp(−L_k/τ) has no "val" superscript), but this cannot be removed entirely.
  - Digits, brackets and word subscripts (img, struct, ens, final, reg, lab, clin) are set upright.
- **Height.** 166 mm. This is within every journal maximum (Nature 247 mm, IEEE 220 mm, PLOS 222 mm). It exceeds the conventions note's ≤ 110 mm suggestion, which that note marks † (an inference). The extra height is the thumbnail strip in (b).
- **Panel (b) maps — real data, recomputed.** No 224² maps exist on disk: `data/processed` is empty, `processed_quick` is 64² with quick-run parameters, and σ is never stored. The script therefore recomputes the maps of one test image at render time (about 0.3 s) using the repository's own code:
  - `tames.fuzzy.fuzzy_maps`, `otsu_binarize`, `dice`, and `tames.data.load_image`, `load_mask`, `nodule_crop`, run on `data/raw/tn3k/test-image/0450.jpg` and its mask.
  - Parameters: those the manuscript reports as selected — α = 2, β = 0.5, γ = 2, λ = 0, polarity "high" (bright side). The main-run `fuzzy_params.json` is not on disk (audit gap), so these come from the manuscript text.
  - SNR₀ = 59.18 from `data/processed_medium/snr0.json` (matches MS §5.1). k = 7, r0 = 2, r_min = 1, r_max = 5 from `configs/default.yaml`.
  - Crop: square, 1.5 × the bounding box of the GT mask (the main experiment's crop), resized to 224² with INTER_AREA (nearest-neighbour for r, Otsu(F) and the mask).
  - Image choice rule, applied once before drawing: Otsu(F)-vs-GT Dice was computed for all 614 test images (mean 0.201, median 0.163). te_0450 is the image closest to the median (Dice 0.163, q = 0.74, native frame 711 × 979). It is representative, not cherry-picked: its Otsu(F) mostly misses the nodule, as the median case does. The scan script and its per-image output are in `work/dice_scan.py` and `work/test_dice.json`.
  - Display: each map is min–max stretched for display, and its range is printed under it. If the raw files are missing, or `--no-maps` is given, panel (b) shows boxes only.
- **Other known residuals (cosmetic).**
  - The "→" glyph in Liberation and Arial is small and sits low. It appears only in "141 → 64 → 1" and "z-score → d̃"; elsewhere words are used.
  - The fusion box is drawn taller than its content because two inputs enter from its left side.

## Checklist — critique P1 (all items)
- [x] **Crop order in lane 1.** The chain is now Input (I_n: B-mode, H × W, 8-bit; M_n: delineation) → fuzzy front end (native H × W) → edge "x̃, F / H × W" → **Crop** (1.5× bbox of M_n, resize to 224²) → edge "[x̃, F] / 224²×2" → stem. M_n has a drawn edge into Crop, labelled "M_n (E1: Otsu(F))".
- [x] **Panel (b) dependency graph.**
  - The c, σ bus goes from Local stats into ATCFM μ, and σ also goes into r, w (Eqs. 5–6).
  - x̃ goes to T, K, q. T, K, q goes into μ. μ goes to ε_w, δ_w, which also receives r, w. Then F (Eq. 8), then Otsu(F).
  - q leaves at Eq. (3) as a blue edge to "q = 0.74 = Q_img in (a)". The outputs are marked "channel 1 of (a)" (x̃) and "channel 2 of (a)" (F); Otsu(F) is marked "E1: crop, d_n".
- [x] **One meaning per stroke.**
  - Solid grey arrow: data flow.
  - Thin solid blue: scalar quality coefficient (the q path, and the Q_struct operand into its × node).
  - Dashed: only the laboratory/clinical/regional box and its arrow (framework-defined, no data in TN3K).
  - Dotted grey: SHAP tap on z and Grad-CAM tap on the layer4 output (evaluation time).
  - All four are in the legend. The black bus is gone.
- [x] **Text overflow.** Checked programmatically in both fonts, with no overflows. The four former overflows no longer exist ([x̃, F] is now an edge label; FC box 12.4 mm; Descriptors two lines; Decision text "t*: sens. ≥ 0.95 (val.)").
- [x] **Width 170 mm.** PNG 4015 px (target 4016 ± 15). PDF 170.00 mm. Max x of any element ≤ 168 mm.
- [x] **Minimum type.** Base sizes 7, 6 and 5.5 pt (plus 8-pt panel letters); nothing smaller. The sub/superscript caveat is above.

## Checklist — critique P2 / P3
- [x] **z-bus detour removed.** z drops from the fusion box to a junction dot and a distribution bar, then into a horizontal row of four learners. Learner outputs go to a collector, then Σ, prior shift, seed mean, decision and a new **Output** terminal (ŷ ∈ {0,1}, p̄_final).
- [x] **No Σ arrowhead blob.** A single collector arrow enters Σ, and the Eq. (14) formula sits beside Σ.
- [x] **Descriptors have an input.** Edge from Input labelled "I_n, M_n (H × W)"; the box says "on x̃ (Eq. 1), M_n".
- [x] **avg pool inside the f_θ container,** with the pool → fc arrow visible (labelled 512).
- [x] **q path** runs above the container, with no white bbox and labels at both ends: "q (Eq. 3)" at its start and "Q_img = q" at the node.
- [x] **Panel (c) dropped.** The caption cites [11] for basic blocks.
- [x] **Legend is complete.**
  - Box keys: fuzzy front end (this work), convolution, dense/pooling, module, input/output (TINT, used only on the two terminals).
  - Symbol keys: operator, data flow, quality coefficient, framework-defined, evaluation-time tap, M_n contour.
  - Titles are black and there is no coloured text. All eight steps of Eqs. (1)–(8) in (b) are blue.
- [x] **Caption-material sentences removed from the drawing:** the ImageNet note, the general-form note, "learners fitted on z…", "frozen on validation", and the (b) footnote.
- [x] **P3 wording.** "Prior shift": Eq. (15) Δ_reg = 0 on TN3K; Eq. (16) p_final = σ(logit p_ens + Δ_reg). Decision is "Eq. (16)". FSVM line reads "s_n: Eqs. (12)–(13)". DL head line reads "end-to-end with f_θ". Q_struct label reads "(= 1 on TN3K)". XGB reads "≤ 600 rounds, logistic / early-stopped on val.".
- [x] **Shapes on edges, one baseline.** The labels inside f_θ are 56²×64, 56²×64, 28²×128, 14²×256, 7²×512 and 512. Further labels sit outside the container: 224²×2, h_img, Q_img h_img ∈ ℝ¹²⁸, d_n ∈ ℝ¹¹, Q_struct d̃ ∈ ℝ¹¹, z ∈ ℝ¹⁴¹, p_k ∈ [0,1], p_ens, p_final, p̄_final.
- [x] **Stem height = stage-1 height** (18 mm, same 56² output). Both × nodes have their operand drawn. The SHAP box is 12.5 mm tall with 6-pt lines, and its tap is detached from every learner edge.
- [x] **z layout drawn** as a segmented vector [128 | 11 | 1 | 1] = 141 under the Eq. (11) formula (U-Net-style shape annotation).

## Checklist — audit discrepancies (figure side)
- [x] **#1 RF.** Now "500 trees, class-balanced / p = mean leaf class fraction / fitted on z". The OOB claim is removed.
- [x] **#2** Native H × W front end → crop/resize; the (b) title says "on the native frame".
- [x] **#3 f_θ container** includes the fc embedding: title "f_θ — Eq. (9): ResNet-18 trunk + fc embedding"; fc box reads "fc, 128 / ReLU, / dropout 0.3".
- [x] **#4 Stem:** "7×7 conv / 64, /2 / BN, ReLU / 3×3 max / pool, /2". Stages read "basic / block ×2 / /2". The shortcut detail (1×1 conv /2 + BN, first block of stages 2–4) moved to the caption, since panel (c) was dropped.
- [x] **#5 DL head:** "141 → 64 → 1, sigmoid / ReLU, dropout 0.3 / end-to-end with f_θ". It is green (dense layers). The other three learners read "fitted on z" (FSVM: "RBF on standardized z").
- [x] **#6 "Seed mean"** box (seeds 0–4) sits between prior shift and decision; the decision uses p̄_final.
- [x] **#7 dashed-style conflicts** resolved (see P1).
- [x] **#9 Eq. (15)/(16) placement** corrected.
- [x] **KQ1:** "widened" wording removed from the figure; the caption uses the correct wording.
- [x] **KQ3:** (b) box renamed from "descriptors" to "T, K, q / texture, SNR"; "λ = 0 selected: F = C (closing)"; Otsu "bright side"; "E1: crop, d_n".
- [x] **KQ3:** q's double use is shown by the Eq. (11) formula and the z segments (q as gate and as a feature).
- [x] **KQ4:** FSVM reads "RBF on standardized z / Platt scaling on val."; XGB is "early-stopped on val.".
- [x] **KQ6:** Grad-CAM taps the stage-4 → avg-pool edge, i.e. the layer4 output. SHAP is "on z: TreeSHAP, KernelSHAP".

## Proposed Figure 2 caption (119 words)

**Figure 2.** Architecture of TAMES AI. (a) Data-flow graph; edges carry per-image tensor shapes. The fuzzy front end runs on the native frame; [x̃, F] is then cropped (1.5× the delineation's bounding box) and resized to 224 × 224. f_θ is ResNet-18 (ImageNet-1k weights; first convolution re-initialised for two channels by averaging the RGB filters; basic blocks as in [11], "/2" = stride 2 with a 1×1-convolution + BN shortcut) plus a 128-unit fc layer. The DL head is trained end-to-end with f_θ; RF, XGBoost and the fuzzy SVM are fitted on z. p_final is averaged over five seeds before t*. (b) Eqs. (1)–(8) with one test image's maps (min–max scaled). Blue: this work; dashed: no data in TN3K; dotted: evaluation-time taps.

## Replacement for the first sentences of Section 4
(Replaces "TAMES AI is organized as eight layers (Figure 2). … the decision-support layer returns label, probability and explanation." Keep the inference-time sentence that follows.)

> Figure 2a shows TAMES AI as a data-flow graph from the native B-mode frame and its delineation to the decision. The fuzzy front end (Sections 3.4–3.5; Figure 2b) computes x̃, F and q on the native frame. The nodule crop of [x̃, F] (1.5× the bounding box, resized to 224 × 224) enters f_θ, a ResNet-18 trunk with a 128-dimensional embedding (Eq. 9), whose output is gated by Q_img = q. The structured branch turns the 11 delineation descriptors into Q_struct d̃ (Eq. 10); laboratory, clinical and regional inputs are defined by the framework but absent from TN3K. The fused vector z ∈ ℝ¹⁴¹ (Eq. 11) feeds four learners: a DL head trained end-to-end with f_θ, and a random forest, XGBoost and a fuzzy SVM fitted on z. Their probabilities are combined by Eq. (14), shifted by the regional prior (Eqs. 15–16), averaged over five seeds and thresholded at t*. SHAP on z and Grad-CAM on the last residual stage of f_θ are computed at evaluation time and returned with the label and probability.

## Manuscript corrections the audit requires
1. **§3.7 learner table, RF row.** "out-of-bag-calibrated class fraction" → "class-fraction probability of a class-balanced forest (mean leaf class fraction over the 500 trees; no calibration)". The code computes OOB probabilities only for the training rows and never uses them; val/test use plain `predict_proba`. (The `rf_train_probability` docstring in `tames/models/learners.py` repeats the claim; code comment only, not changed.)
2. **Order of operations and the meaning of H × W** (Table 2, §3.3 notation, §3.13). State that Eqs. (1)–(8), q and the 11 descriptors are computed on the native H × W frame, and that the 1.5× crop and the resize to 224 × 224 follow. Table 2's "H × W = 224 × 224" is the network input size, not the H × W of Eqs. (1)–(8).
3. **§3.6 / §3.13 backbone wording.** "first convolution widened to two input channels by averaging the pretrained filters" → "first convolution re-initialised for two input channels: the ImageNet RGB filters are averaged and copied to both channels" (3 → 2 channels; nothing is widened).
4. **Eq. (10) / Eq. (11) modality set.**
   - Add "struct" to Eq. (10)'s set m ∈ {lab, clin, reg}, since Q_struct uses Eq. (10) on TN3K.
   - Add the descriptor block Q_struct d̃ to the general Eq. (11), so that d_z = d_h + d_d + d_l + d_c + d_r + 5. That reduces to 128 + 11 + 2 = 141 on TN3K.
5. **§3.7 DL row.** "tuned: learning rate, weight decay, dropout, augmentation strength" → "fixed a priori (AdamW, lr 3·10⁻⁴, weight decay 10⁻⁴, dropout 0.3, fixed augmentation)". Keep "tuned" only if the tuning was done outside the released code.
6. **§4 and the Figure 2 caption.** Replace the "eight layers" text and caption with the versions above ("highlighted box" → "blue").
7. **Descriptor rule** (Table 2 / §3.x). "all NaN when the delineation is empty" → "when the delineation has fewer than 10 pixels".
8. **Optional, §3.7 XGBoost row.** Add "number of rounds by early stopping on validation log-loss (patience 30, at most 600)". The code uses `eval_metric="logloss"`, `early_stopping_rounds=30` and `eval_set` = validation, so the figure's "early-stopped on val." needs this sentence to be traceable in the text.
9. **Not manuscript, for the repo.** The `pipeline.py` header says "out-of-fold embeddings for the training rows", but the code trains one in-sample model per seed. This is a stale docstring.
