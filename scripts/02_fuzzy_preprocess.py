#!/usr/bin/env python
"""Step 2 — ATCFM + AFMO maps, nodule crops, structured descriptors, segmentation Dice.

Stage A (tuning, validation subset): α, β, γ, λ and the Otsu polarity are selected by
the mean Dice between the Otsu-binarized AFMO map F and the reference masks
(Section 3.5). The baseline is Otsu on the raw normalized image.
Stage B (all images): with the selected parameters, every image is turned into a
nodule-centered crop with channels [x̃, μ, F, F_plain] (F_plain = AFMO applied to the
plain Gaussian membership, for ablation b), plus q, SNR, per-image Dice and the
structured descriptors.

Outputs under <processed>/: fuzzy_params.json, X_{train,val,test}.npy (fp16),
meta_{train,val,test}.csv.

    python scripts/02_fuzzy_preprocess.py --config configs/default.yaml [--workers 8] [--no-tune]
"""
import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tames.config import load_config  # noqa: E402
from tames.data import load_image, load_mask, nodule_crop  # noqa: E402
from tames.features import DESCRIPTOR_NAMES, sonographic_descriptors  # noqa: E402
from tames.fuzzy import afmo, dice, fuzzy_maps, otsu_binarize  # noqa: E402
from tames.fuzzy.atcfm import atcfm  # noqa: E402

_CFG = {}


def _init(cfg):
    _CFG.update(cfg)


# ------------------------------------------------------------------ stage A ---
def _tune_one(args):
    """For one image: Dice of Otsu(F) for every (α, β, γ) × λ × polarity, plus the raw baseline."""
    row, grid, fz, snr0 = args
    img, mask = load_image(row["image_path"]), load_mask(row["mask_path"])
    res = {}
    x = None
    for a, b, g in itertools.product(grid["alpha"], grid["beta"], grid["gamma"]):
        p = dict(fz, alpha=a, beta=b, gamma=g)
        m = fuzzy_maps(img, p, snr0)
        x = m["x"]
        O, C = m["O"], m["C"]
        for lam in grid["lam"]:
            F = lam * O + (1 - lam) * C
            for pol in ("low", "high"):
                res[(a, b, g, lam, pol)] = dice(otsu_binarize(F, pol), mask)
    base = {pol: dice(otsu_binarize(x, pol), mask) for pol in ("low", "high")}
    return res, base


def tune(df, cfg, workers):
    fz, grid, snr0 = cfg["fuzzy"], cfg["fuzzy"]["grid"], cfg["_snr0"]
    val = df[df.split == "val"]
    n = min(int(fz.get("tune_subset", len(val))), len(val))
    sub = val.iloc[np.random.default_rng(int(cfg["seed"])).permutation(len(val))[:n]]
    jobs = [(r, grid, fz, snr0) for _, r in sub.iterrows()]
    acc, base = {}, {"low": [], "high": []}
    with Pool(workers) as pool:
        for res, b in tqdm(pool.imap_unordered(_tune_one, jobs), total=len(jobs), desc="tune α,β,γ,λ"):
            for k, v in res.items():
                acc.setdefault(k, []).append(v)
            for pol in base:
                base[pol].append(b[pol])
    means = {k: float(np.mean(v)) for k, v in acc.items()}
    (a, b, g, lam, pol), best = max(means.items(), key=lambda kv: kv[1])
    raw = {pol: float(np.mean(v)) for pol, v in base.items()}
    table = [{"alpha": k[0], "beta": k[1], "gamma": k[2], "lam": k[3], "polarity": k[4], "dice": v}
             for k, v in means.items()]
    return {"alpha": a, "beta": b, "gamma": g, "lam": lam, "polarity": pol, "val_dice": best,
            "val_dice_raw_otsu": raw, "n_tune": n, "table": table}


# ------------------------------------------------------------------ stage B ---
def _process_one(args):
    row, fz, snr0, size, margin, mask_source, polarity = args
    img = load_image(row["image_path"])
    mask_gt = load_mask(row["mask_path"])
    m = fuzzy_maps(img, fz, snr0)                       # ATCFM + AFMO (Eqs. 1–8)
    mu_plain = atcfm(m["x"], m["c"], m["sigma"], m["T"], m["K"], 1.0, 0.0, 0.0)
    F_plain = afmo(mu_plain, m["sigma"], m["sigma_g"], fz["r0"], fz["gamma"],
                   int(fz["r_min"]), int(fz["r_max"]), fz["lam"])["F"]
    seg = otsu_binarize(m["F"], polarity)
    d_afmo = dice(seg, mask_gt)
    d_raw = dice(otsu_binarize(m["x"], polarity), mask_gt)
    mask = mask_gt if mask_source == "gt" else seg
    y0, y1, x0, x1 = nodule_crop(mask_gt if mask_source == "gt" else seg, margin)
    chans = []
    for ch in (m["x"], m["mu"], m["F"], F_plain):
        chans.append(cv2.resize(ch[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA))
    X = np.stack(chans).astype(np.float16)
    desc = sonographic_descriptors(m["x"], mask)
    meta = {"id": row["id"], "split": row["split"], "label": int(row["label"]), "q": m["q"],
            "snr": m["snr"], "dice_afmo": d_afmo, "dice_raw": d_raw, "mask_empty": int(mask.sum() == 0),
            "crop": f"{y0},{y1},{x0},{x1}", **desc}
    return X, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-tune", action="store_true", help="use α, β, γ, λ from the config as they are")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    proc = Path(cfg["paths"]["processed"])
    df = pd.read_csv(proc / "splits.csv")
    cfg["_snr0"] = json.load(open(proc / "snr0.json"))["snr0"]
    fz = dict(cfg["fuzzy"])
    polarity = "low"
    if not args.no_tune:
        sel = tune(df, cfg, args.workers)
        fz.update({k: sel[k] for k in ("alpha", "beta", "gamma", "lam")})
        polarity = sel["polarity"]
        json.dump({**{k: sel[k] for k in ("alpha", "beta", "gamma", "lam", "polarity", "val_dice",
                                          "val_dice_raw_otsu", "n_tune")},
                   "k": fz["k"], "r0": fz["r0"], "r_min": fz["r_min"], "r_max": fz["r_max"],
                   "snr0": cfg["_snr0"], "grid_table": sel["table"]},
                  open(proc / "fuzzy_params.json", "w"), indent=2)
        print(f"selected α={sel['alpha']} β={sel['beta']} γ={sel['gamma']} λ={sel['lam']} polarity={polarity}"
              f"  val Dice {sel['val_dice']:.3f} (raw Otsu {sel['val_dice_raw_otsu'][polarity]:.3f})")
    else:
        json.dump({**{k: fz[k] for k in ("alpha", "beta", "gamma", "lam", "k", "r0", "r_min", "r_max")},
                   "polarity": polarity, "snr0": cfg["_snr0"], "tuned": False},
                  open(proc / "fuzzy_params.json", "w"), indent=2)
    im = cfg["image"]
    for split in ("train", "val", "test"):
        rows = df[df.split == split]
        jobs = [(r, fz, cfg["_snr0"], int(im["size"]), float(im["crop_margin"]), im["mask_source"], polarity)
                for _, r in rows.iterrows()]
        Xs, metas = [], []
        with Pool(args.workers) as pool:
            for X, meta in tqdm(pool.imap(_process_one, jobs), total=len(jobs), desc=f"maps {split}"):
                Xs.append(X); metas.append(meta)
        np.save(proc / f"X_{split}.npy", np.stack(Xs))
        pd.DataFrame(metas).to_csv(proc / f"meta_{split}.csv", index=False)
        md = pd.DataFrame(metas)
        print(f"{split}: n={len(md)}  Dice AFMO {md.dice_afmo.mean():.3f} vs raw Otsu {md.dice_raw.mean():.3f}"
              f"  empty masks {int(md.mask_empty.sum())}  q mean {md.q.mean():.3f}")
    print("descriptors:", DESCRIPTOR_NAMES)


if __name__ == "__main__":
    main()
