#!/usr/bin/env python
"""Step 1 — build the split table and the image-quality reference SNR_0.

Writes <processed>/splits.csv (id, split, image_path, mask_path, label, snr, H, W)
and <processed>/snr0.json (median SNR of the training images; Eq. 3).

    python scripts/01_prepare_data.py --config configs/default.yaml
    python scripts/01_prepare_data.py --config configs/default.yaml --verify-labels /path/to/official_csvs
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tames.config import load_config  # noqa: E402
from tames.data import load_image, load_split  # noqa: E402
from tames.fuzzy import normalize, snr  # noqa: E402


def verify_labels(labels_dir: Path, official_dir: Path) -> None:
    for name in ("label4trainval.csv", "label4test.csv"):
        a = dict(l.strip().split(",") for l in (labels_dir / name).read_text().strip().splitlines())
        b = dict(l.strip().split(",") for l in (official_dir / name).read_text().strip().splitlines())
        diff = [k for k in a if a[k] != b.get(k)]
        missing = [k for k in b if k not in a]
        print(f"{name}: {len(a)} rows here, {len(b)} official, {len(diff)} differ, {len(missing)} missing here")
        if diff:
            print("  first differences:", diff[:10])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. fold=1")
    ap.add_argument("--verify-labels", default=None, help="directory holding the official ACL label CSVs")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    paths = cfg["paths"]
    if args.verify_labels:
        verify_labels(Path(paths["labels"]), Path(args.verify_labels))
        return
    df = load_split(paths["raw"], paths["labels"], fold=int(cfg["fold"]))
    if cfg.get("subset"):
        rng = np.random.default_rng(int(cfg["seed"]))
        parts = []
        for split, g in df.groupby("split"):
            parts.append(g.iloc[rng.permutation(len(g))[: int(cfg["subset"])]])
        df = pd.concat(parts).reset_index(drop=True)
    snrs, hs, ws = [], [], []
    for p in tqdm(df["image_path"], desc="SNR"):
        img = load_image(p)
        hs.append(img.shape[0]); ws.append(img.shape[1])
        snrs.append(snr(normalize(img)))
    df["snr"], df["H"], df["W"] = snrs, hs, ws
    out = Path(paths["processed"]); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "splits.csv", index=False)
    snr0 = float(np.median(df.loc[df.split == "train", "snr"]))
    json.dump({"snr0": snr0}, open(out / "snr0.json", "w"), indent=2)
    print(df.groupby("split")["label"].agg(["count", "mean"]).rename(columns={"mean": "malignant_frac"}))
    print(f"SNR_0 (median training SNR) = {snr0:.3f}; image sizes H {min(hs)}–{max(hs)}, W {min(ws)}–{max(ws)}")
    print("wrote", out / "splits.csv")


if __name__ == "__main__":
    main()
