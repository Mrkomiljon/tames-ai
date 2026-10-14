#!/usr/bin/env bash
# Full TN3K experiment: fold-0 train/val, official test. ~2–4 h on an Apple-Silicon Mac (MPS)
# or a mid-range GPU; CPU-only is several times slower.
#   bash scripts/run_all.sh                      # default config, seeds 0–4
#   bash scripts/run_all.sh configs/quick.yaml   # smoke test (minutes, CPU)
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-configs/default.yaml}"
WORKERS="${WORKERS:-$(python -c 'import os; print(max(1, os.cpu_count() - 1))')}"
python scripts/01_prepare_data.py   --config "$CFG"
python scripts/02_fuzzy_preprocess.py --config "$CFG" --workers "$WORKERS"
python scripts/03_train_cnn.py      --config "$CFG" --variant tames
python scripts/03_train_cnn.py      --config "$CFG" --variant image_only
python scripts/04_train_learners.py --config "$CFG" --variant tames
python scripts/05_ensemble_eval.py  --config "$CFG" --variant tames
python scripts/06_ablation.py       --config "$CFG"
python scripts/07_explain.py        --config "$CFG" --variant tames
echo "done → runs/$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['run_name'])")"
