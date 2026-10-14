#!/usr/bin/env python
"""Step 8 — image-quality robustness: AUC of TAMES vs the image-only CNN and the late-fusion
baseline under additive/speckle noise, blur and contrast loss applied to the raw test images
(front end recomputed, models frozen). Needs steps 03–05 for --variant tames and image_only.

Writes runs/<run>/robustness/{robustness.csv, robustness_summary.csv, robustness.png}.

    python scripts/08_robustness.py --config configs/default.yaml --seeds 0 1 2 3 4
"""
from _common import finish, parse
from tames.robustness import run


def main():
    ap = parse(__doc__, variant=False)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-images", type=int, default=None, help="debug: use only the first N test images")
    args = ap.parse_args()
    cfg, seeds = finish(args)
    run(cfg, list(seeds), n_boot=args.n_boot, workers=args.workers, max_images=args.max_images)


if __name__ == "__main__":
    main()
