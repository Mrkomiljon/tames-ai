#!/usr/bin/env python
"""Step 4 — build z (Eqs. 10–11) and fit RF, XGBoost, Fuzzy SVM and the structured-only
XGBoost baseline; write probabilities to runs/<run>/learners_<variant>_seed<k>/.

    python scripts/04_train_learners.py --config configs/default.yaml --variant tames --seeds 0 1 2 3 4
"""
from _common import finish, parse
from tames.pipeline import learners_stage


def main():
    args = parse(__doc__).parse_args()
    cfg, seeds = finish(args)
    for seed in seeds:
        learners_stage(cfg, args.variant, seed=seed, quality_on=not args.quality_off)


if __name__ == "__main__":
    main()
