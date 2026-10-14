#!/usr/bin/env python
"""Step 3 — train f_θ and the DL learner; write out-of-fold embeddings (train) and
final-model embeddings/probabilities (val, test) to runs/<run>/cnn_<variant>_seed<k>/.

    python scripts/03_train_cnn.py --config configs/default.yaml --variant tames --seeds 0 1 2 3 4
    python scripts/03_train_cnn.py --config configs/default.yaml --variant image_only
"""
from _common import finish, parse
from tames.pipeline import cnn_stage


def main():
    args = parse(__doc__).parse_args()
    cfg, seeds = finish(args)
    for seed in seeds:
        cnn_stage(cfg, args.variant, seed=seed, quality_on=not args.quality_off)


if __name__ == "__main__":
    main()
