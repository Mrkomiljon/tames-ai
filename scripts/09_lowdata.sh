#!/usr/bin/env bash
# Low-data experiment: retrain TAMES and the image-only CNN on 10 %, 25 % and 50 % of the
# training partition (stratified, seeded), validation/test unchanged. Each fraction is its
# own run directory runs/tn3k_frac<pct>/. Afterwards scripts/10_lowdata_table.py builds the table.
#   bash scripts/09_lowdata.sh                 # fractions 0.1 0.25 0.5, seeds from the config
#   SEEDS="0 1 2" bash scripts/09_lowdata.sh   # fewer seeds
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${CFG:-configs/default.yaml}"
SEEDARGS=""; [ -n "${SEEDS:-}" ] && SEEDARGS="--seeds $SEEDS"
for FR in ${FRACS:-0.1 0.25 0.5}; do
  PCT=$(python -c "print(int(round($FR*100)))")
  RUN="tn3k_frac${PCT}"
  echo "=== train_frac=$FR → runs/$RUN ==="
  python scripts/03_train_cnn.py      --config "$CFG" --variant tames      --set train_frac=$FR run_name=$RUN $SEEDARGS
  python scripts/03_train_cnn.py      --config "$CFG" --variant image_only --set train_frac=$FR run_name=$RUN $SEEDARGS
  python scripts/04_train_learners.py --config "$CFG" --variant tames      --set train_frac=$FR run_name=$RUN $SEEDARGS
  python scripts/05_ensemble_eval.py  --config "$CFG" --variant tames      --set train_frac=$FR run_name=$RUN $SEEDARGS
done
python scripts/10_lowdata_table.py --config "$CFG"
