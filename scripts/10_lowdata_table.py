#!/usr/bin/env python
"""Collect the low-data runs into one table + figure: test AUC (CI) of TAMES, image-only CNN,
late fusion and descriptor-only XGB at 10/25/50/100 % of the training data, with the
paired DeLong p of TAMES vs image-only at each fraction.
Writes runs/lowdata_table.csv and runs/lowdata.png.
"""
import json
from pathlib import Path

import pandas as pd
from _common import finish, parse
from tames.config import load_config


def main():
    ap = parse(__doc__, variant=False, seeds=False)
    args = ap.parse_args()
    cfg, _ = finish(args)
    runs = Path(cfg["paths"]["runs"])
    rows = []
    for pct, run in ((10, "tn3k_frac10"), (25, "tn3k_frac25"), (50, "tn3k_frac50"), (100, cfg["run_name"])):
        f = runs / run / "ensemble_tames_seed0" / "results.json"
        if not f.exists():
            print("missing", f); continue
        r = json.load(open(f))
        comp = r.get("comparisons_vs_tames", {})
        for m in ("tames", "cnn_image_only", "late_fusion_cnn_xgb", "xgb_struct_only", "dl_only"):
            if m not in r["models"]:
                continue
            t, ci = r["models"][m]["test"], r["models"][m]["test_ci"]
            rows.append({"train_pct": pct, "model": m, "auc": t["auc"], "ci_low": ci["auc"][0], "ci_high": ci["auc"][1],
                         "auc_seed_sd": r["models"][m]["test_auc_seed_sd"],
                         "delong_p_vs_tames": comp.get(m, {}).get("delong", {}).get("p"),
                         "brier": r["models"][m]["calibration_test"]["brier"]})
    df = pd.DataFrame(rows)
    df.to_csv(runs / "lowdata_table.csv", index=False)
    print(df.to_string(index=False))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for m in ("tames", "cnn_image_only", "late_fusion_cnn_xgb", "xgb_struct_only"):
            d = df[df.model == m].sort_values("train_pct")
            if len(d):
                ax.errorbar(d.train_pct, d.auc, yerr=[d.auc - d.ci_low, d.ci_high - d.auc], marker="o", capsize=3, label=m)
        ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100]); ax.set_xticklabels(["10 %", "25 %", "50 %", "100 %"])
        ax.set_xlabel("fraction of training images"); ax.set_ylabel("test AUC (95 % CI)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(runs / "lowdata.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
