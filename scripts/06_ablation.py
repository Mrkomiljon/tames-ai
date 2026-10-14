#!/usr/bin/env python
"""Step 6 — ablation (Section 3.12): one component removed at a time, the rest re-tuned.

    (a) AFMO off          CNN variant abl_afmo_off  (F replaced by μ)         + learners + ensemble
    (b) ATCFM off         CNN variant abl_atcfm_off (plain Gaussian, q = 1)   + learners + ensemble
    (c) quality off       tames CNN + learners with every Q_m = 1 (--quality-off)
    (d) regional off      Δ_reg = 0 and the regional vector removed — on TN3K the regional inputs
                          are absent and Δ_reg = 0 already, so (d) equals the full model (reported as such)
    (e) equal weights     ensemble with τ → ∞
    (f) drop each learner ensemble without dl / rf / xgb / fsvm

Assumes steps 03–05 were run for --variant tames (and image_only). Writes
runs/<run>/ablation_table.csv with test AUC, ΔAUC vs the full model and a paired
bootstrap CI of the difference.

    python scripts/06_ablation.py --config configs/default.yaml --seeds 0 1 2 3 4
"""
import json

import numpy as np
import pandas as pd
from _common import finish, parse
from tames.config import run_dir
from tames.eval import bootstrap_metric_difference, delong_test
from tames.pipeline import (cnn_stage, ensemble_stage, learners_stage, load_processed,
                            stage_name)


def _pooled_test(cfg, tag):
    d = run_dir(cfg) / tag
    order = json.load(open(d / "model_order.json"))
    P = np.load(d / "p_pooled_test.npy")
    r = json.load(open(d / "results.json"))
    return P[order.index("tames")], r["models"]["tames"]["threshold"]["t_star"], r["models"]["tames"]["test"]["auc"]


def main():
    ap = parse(__doc__, variant=False)
    ap.add_argument("--skip-training", action="store_true", help="only re-run the ensemble ablations (e), (f)")
    args = ap.parse_args()
    cfg, seeds = finish(args)
    y = load_processed(cfg)["y"]["test"]
    n_boot = int(cfg["eval"]["n_boot"])
    full_tag = stage_name("ensemble", "tames", 0, True)
    p_full, t_full, auc_full = _pooled_test(cfg, full_tag)
    rows = [{"ablation": "none (full TAMES AI)", "test_auc": auc_full, "delta_auc": 0.0, "ci_low": 0.0,
             "ci_high": 0.0, "delong_p": None}]

    def add(name, tag):
        p, t, auc = _pooled_test(cfg, tag)
        d = bootstrap_metric_difference(y, p, p_full, t, t_full, "auc", n_boot)
        rows.append({"ablation": name, "test_auc": auc, "delta_auc": d["diff"], "ci_low": d["ci_low"],
                     "ci_high": d["ci_high"], "delong_p": delong_test(y, p, p_full)["p"]})
        print(f"  {name:38s} AUC {auc:.3f}  ΔAUC {d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")

    if not args.skip_training:
        for name, variant in (("(a) AFMO off (F → μ)", "abl_afmo_off"), ("(b) ATCFM off (plain Gaussian)", "abl_atcfm_off")):
            for seed in seeds:
                cnn_stage(cfg, variant, seed=seed)
                learners_stage(cfg, variant, seed=seed)
            ensemble_stage(cfg, variant, seeds=seeds, tag=f"ablation_{variant}")
            add(name, f"ablation_{variant}")
        for seed in seeds:
            cnn_stage(cfg, "tames", seed=seed, quality_on=False)
            learners_stage(cfg, "tames", seed=seed, quality_on=False)
        ensemble_stage(cfg, "tames", seeds=seeds, quality_on=False, tag="ablation_quality_off")
        add("(c) quality coefficients off (Q_m = 1)", "ablation_quality_off")
    rows.append({"ablation": "(d) regional inputs off", "test_auc": auc_full, "delta_auc": 0.0, "ci_low": 0.0,
                 "ci_high": 0.0, "delong_p": None,
                 "note": "TN3K has no regional inputs and Δ_reg = 0; identical to the full model"})
    ensemble_stage(cfg, "tames", seeds=seeds, equal_weights=True, tag="ablation_equal_weights")
    add("(e) equal ensemble weights (τ → ∞)", "ablation_equal_weights")
    for k in ("dl", "rf", "xgb", "fsvm"):
        ensemble_stage(cfg, "tames", seeds=seeds, drop=k, tag=f"ablation_drop_{k}")
        add(f"(f) without {k.upper()} learner", f"ablation_drop_{k}")
    df = pd.DataFrame(rows)
    df.to_csv(run_dir(cfg) / "ablation_table.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
