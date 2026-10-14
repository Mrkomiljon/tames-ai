#!/usr/bin/env python
"""Step 5 — adaptive ensemble (Eq. 14), regional prior (Eq. 15), decision rule (Eq. 16),
baselines, metrics with bootstrap CIs, DeLong and McNemar tests, calibration.
Writes runs/<run>/ensemble_<variant>_seed0/{results.json, results_table.csv}.

    python scripts/05_ensemble_eval.py --config configs/default.yaml --variant tames --seeds 0 1 2 3 4
    python scripts/05_ensemble_eval.py ... --equal-weights          # ablation (e)
    python scripts/05_ensemble_eval.py ... --drop fsvm              # ablation (f)
"""
from _common import finish, parse
from tames.pipeline import ensemble_stage


def main():
    ap = parse(__doc__)
    ap.add_argument("--equal-weights", action="store_true")
    ap.add_argument("--drop", default=None, choices=[None, "dl", "rf", "xgb", "fsvm"])
    args = ap.parse_args()
    cfg, seeds = finish(args)
    ensemble_stage(cfg, args.variant, seeds=seeds, quality_on=not args.quality_off,
                   equal_weights=args.equal_weights, drop=args.drop)


if __name__ == "__main__":
    main()
