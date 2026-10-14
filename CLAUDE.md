# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Reference implementation of the manuscript *TAMES AI: An Adaptive Fuzzy, Multimodal Ensemble
Model for Malignancy Risk Assessment of Thyroid Nodules*, run on the TN3K ultrasound dataset.
Every equation of Section 3 maps to one function; the README has the full Eq. → code table and
the design decisions that must be stated in the paper. Read the README before changing
anything that affects reported numbers. Not a git repository.

## Commands

```bash
/opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
brew install libomp unar                    # OpenMP runtime for XGBoost; unar replaces unrar
bash scripts/00_download_tn3k.sh            # needs git + unrar or unar; fills data/raw/tn3k
python -m pytest tests -q                   # 11 range/reduction checks of Eqs. 1–8, 13, 14
python -m pytest tests/test_fuzzy.py -q -k afmo_convex   # single test

bash scripts/run_all.sh configs/quick.yaml  # smoke test: 160 images, 64 px, minutes on CPU
bash scripts/run_all.sh                     # full run (configs/default.yaml), seeds 0–4, hours
```

Steps individually (always from the repo root; `--config` defaults to `configs/default.yaml`
and relative paths resolve against the repo root via `tames/config.py`):

```bash
python scripts/01_prepare_data.py   --config configs/quick.yaml
python scripts/02_fuzzy_preprocess.py --config configs/quick.yaml --workers 8 [--no-tune]
python scripts/03_train_cnn.py      --config configs/quick.yaml --variant tames --seeds 0 1
python scripts/04_train_learners.py --config configs/quick.yaml --variant tames
python scripts/05_ensemble_eval.py  --config configs/quick.yaml --variant tames [--equal-weights] [--drop fsvm]
python scripts/06_ablation.py       --config configs/quick.yaml [--skip-training]
python scripts/07_explain.py        --config configs/quick.yaml --variant tames --seeds 0
```

`--set key=value ...` overrides any dotted config entry on every script (values are parsed
as YAML, e.g. `--set cnn.epochs=5 paths.processed=data/processed_ddti`).

## macOS environment (Apple Silicon)

- Use Homebrew's Python 3.12 for the venv. The pyenv 3.11.9 build lacks `_lzma`, which
  torchvision imports, so pytest fails at collection under it.
- Homebrew no longer ships `unrar` and the `rar` cask is disabled; the download script
  falls back to `unar`.
- **XGBoost segfaults after torch is imported** (step 04) because XGBoost links Homebrew's
  `libomp.dylib` while torch bundles its own, so two OpenMP runtimes load. The fix applied
  to this venv repoints XGBoost's rpath at torch's lib directory and re-signs the dylib:

  ```bash
  LIB=.venv/lib/python3.12/site-packages/xgboost/lib/libxgboost.dylib
  install_name_tool -rpath /opt/homebrew/opt/libomp/lib "$PWD/.venv/lib/python3.12/site-packages/torch/lib" "$LIB"
  codesign -f -s - "$LIB"
  ```

  Reinstalling or upgrading xgboost undoes this; `KMP_DUPLICATE_LIB_OK=TRUE` does not help.

## Architecture

**Config → run directory.** `load_config` (YAML + overrides) feeds every stage. Outputs go to
`runs/<run_name>/` and preprocessed arrays to `paths.processed`. `quick.yaml` and
`default.yaml` differ in `run_name`, `paths.processed` and `subset`, so smoke runs never
overwrite real ones.

**Two layers.** `scripts/0N_*.py` are thin CLIs; `scripts/_common.py` gives steps 03–07 a
shared parser (`--config`, `--set`, `--variant`, `--seeds`, `--quality-off`). All logic for
steps 03–06 lives in `tames/pipeline.py` as `cnn_stage`, `learners_stage`, `ensemble_stage`,
which `06_ablation.py` calls directly in a loop. Step 02 keeps its logic in the script
(multiprocessing over images, calling `tames/fuzzy/pipeline.py: fuzzy_maps`).

**Stage chaining by directory name.** Stages find each other's outputs through
`stage_name(kind, variant, seed, quality_on)` → `cnn_tames_seed0`, `learners_tames_qoff_seed1`,
`ensemble_tames_seed0`, etc. The ensemble stage also silently picks up `cnn_image_only_seed<k>`
if it exists to add the image-only and late-fusion baselines, so step 03 must be run for
both `tames` and `image_only` before step 05 for a complete Table 3. Ablations write to
`ablation_<name>` tags instead.

**Channel layout and variants.** `X_<split>.npy` from step 02 is fp16 with channels
`[x̃, μ, F, F_plain]`. `VARIANTS` in `tames/pipeline.py` selects channels per variant:
`tames=[0,2]`, `image_only=[0]`, `abl_afmo_off=[0,1]`, `abl_atcfm_off=[0,3]`. `image_only`
is the only variant that also drops the tabular input to the CNN.

**Fusion vector z** (`tames/features/fusion.py`): `[Q_img·h_img ; Q_struct·s̃ ; Q_img ; Q_struct]`.
Feature names in `learner_params.json` / `learners.joblib` follow this order
(`h_img_0..`, descriptor names, `Q_img`, `Q_struct`); `07_explain.py` and the
structured-only baseline rely on the `h_img_` prefix to split the block.

**Train/val/test discipline.** Everything is fit on `train` only (CNN weights,
`TabularPreprocessor` statistics, learners); hyperparameters, τ, and the thresholds t\*
(sens ≥ `ensemble.min_sensitivity`) and Youden are chosen on `val`; `test` is scored once.
Training-row embeddings are in-sample from the single final CNN per seed, which is why
learners are tuned on validation only. Multi-seed results are pooled by averaging
probabilities across seeds before metrics; per-seed AUCs give the seed SD.

**Fuzzy stage.** `tames/fuzzy/atcfm.py` (Eqs. 1–4) and `afmo.py` (Eqs. 5–8) are pure
NumPy on normalized `[0,1]` float32 maps. α, β, γ, λ and Otsu polarity are re-selected by
Dice on `fuzzy.tune_subset` validation images in step 02, overriding the values in the
config block (`fuzzy_params.json` records the choice).

## Gotchas

- The module docstrings of `tames/pipeline.py` and `scripts/03_train_cnn.py` still mention
  "out-of-fold embeddings"; the code and README decision 2 use in-sample embeddings. Trust
  the code and the README.
- `data/labels/*.csv` are rebuilt from a third-party manifest. Run
  `python scripts/01_prepare_data.py --verify-labels <dir>` against the official ACL CSVs
  before any numbers are published.
- Ablation (d) (regional inputs) is a copy of the full model on TN3K because Δ_reg = 0;
  `ensemble.pi_reg: null` keeps it that way.
- `device: auto` picks MPS on Apple Silicon. Step 02 uses `multiprocessing.Pool` with a
  module-level config dict, so new per-image parameters must be passed through `_CFG`.
- Tests only cover the fuzzy equations, FSVM membership and ensemble weights; there are no
  tests for the data loaders or the stage functions. Use `configs/quick.yaml` as the
  integration check.
