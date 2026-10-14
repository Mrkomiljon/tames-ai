"""Image-quality robustness test (supports the 'uncertainty-aware' claim with numbers).

Each test image is degraded at the *raw* level (noise, blur, contrast loss), then the
whole TAMES front end is recomputed exactly as in step 2 — fuzzy maps, q, crop,
descriptors — so the quality coefficient and the AFMO channel react to the degradation.
The frozen models (CNN, learners, ensemble weights, t*) are then scored. The comparison
is the AUC curve of TAMES vs the image-only CNN (and the late-fusion baseline) as the
degradation strengthens: a smaller drop for TAMES is the evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

from .config import run_dir
from .data import load_image, load_mask, nodule_crop
from .eval import bootstrap_ci, classification_metrics
from .features import DESCRIPTOR_NAMES, fuse, quality_from_indicators, sonographic_descriptors
from .fuzzy import fuzzy_maps
from .models import ensemble_probability, final_probability, get_device
from .models.cnn import load_model, predict
from .pipeline import VARIANTS, load_processed, stage_name, tabular_block

DEGRADATIONS = {
    "gaussian_noise": [0.05, 0.10, 0.20],   # σ of additive noise on the [0, 255]/255 scale
    "speckle_noise": [0.10, 0.20, 0.40],    # σ of multiplicative noise  I·(1+ε)
    "blur": [1.0, 2.0, 4.0],                # Gaussian blur σ in pixels
    "contrast": [0.7, 0.5, 0.3],            # contrast scale around the image mean
}


def degrade(img: np.ndarray, kind: str, level: float, rng: np.random.Generator) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    if kind == "gaussian_noise":
        x = x + rng.normal(0.0, level, x.shape)
    elif kind == "speckle_noise":
        x = x * (1.0 + rng.normal(0.0, level, x.shape))
    elif kind == "blur":
        x = cv2.GaussianBlur(x, (0, 0), level)
    elif kind == "contrast":
        m = float(x.mean())
        x = m + level * (x - m)
    else:
        raise ValueError(kind)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _front_end(img, mask, fz, snr0, size, margin, chans):
    """Mirror of scripts/02_fuzzy_preprocess.py for one (possibly degraded) image."""
    m = fuzzy_maps(img, fz, snr0)
    y0, y1, x0, x1 = nodule_crop(mask, margin)
    maps = {0: m["x"], 1: m["mu"], 2: m["F"]}
    X = np.stack([cv2.resize(maps[c][y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
                  for c in chans]).astype(np.float32)
    desc = sonographic_descriptors(m["x"], mask)
    return X, np.array([desc[k] for k in DESCRIPTOR_NAMES], dtype=np.float64), float(m["q"])


def _job(args):
    path, mask, kind, level, seed, fz, snr0, size, margin, chans = args
    img = load_image(path)
    if kind != "clean":
        img = degrade(img, kind, level, np.random.default_rng(seed))
    return _front_end(img, mask, fz, snr0, size, margin, chans)


def run(cfg: dict, seeds, n_boot: int = 500, out_name: str = "robustness", workers: int = 4,
        max_images: int | None = None, with_quality_off: bool = True, log=print) -> pd.DataFrame:
    """`with_quality_off=True` also scores the ablation-(c) models (every Q_m = 1), when they
    exist, as 'tames_quality_off' — this isolates whether the multiplicative quality gating of
    Eq. (11) is what makes TAMES fragile under degradation."""
    from multiprocessing import Pool
    proc = Path(cfg["paths"]["processed"])
    splits = pd.read_csv(proc / "splits.csv")
    test = splits[splits.split == "test"].reset_index(drop=True)
    if max_images:
        test = test.iloc[:int(max_images)].reset_index(drop=True)
    fp = json.load(open(proc / "fuzzy_params.json"))
    fz = {k: fp[k] for k in ("k", "alpha", "beta", "r0", "gamma", "r_min", "r_max", "lam")}
    snr0 = float(fp["snr0"])
    im = cfg["image"]
    size, margin = int(im["size"]), float(im["crop_margin"])
    device = get_device(cfg.get("device", "auto"))
    data = load_processed(cfg)
    tab = tabular_block(data)                       # preprocessor fitted on train (unchanged)
    y = test["label"].values.astype(int)
    # frozen pieces per seed
    ens = json.load(open(run_dir(cfg) / stage_name("ensemble", "tames", 0) / "results.json"))
    t_star = {m: ens["models"][m]["threshold"]["t_star"] for m in ens["models"]}
    pieces = []
    for s in seeds:
        w = {k: float(v) for k, v in ens["per_seed"][seeds.index(s)]["weights"].items()}
        L = joblib.load(run_dir(cfg) / stage_name("learners", "tames", s) / "learners.joblib")
        cnn_t = load_model(cfg, 2, run_dir(cfg) / stage_name("cnn", "tames", s) / "model.pt", device,
                           tab["test"]["tab"].shape[1]).eval()
        cnn_i = load_model(cfg, 1, run_dir(cfg) / stage_name("cnn", "image_only", s) / "model.pt", device).eval()
        pieces.append((w, L, cnn_t, cnn_i))
    qoff = []
    qoff_dir = run_dir(cfg) / "ablation_quality_off"
    if with_quality_off and (qoff_dir / "results.json").exists():
        ens_q = json.load(open(qoff_dir / "results.json"))
        t_star["tames_quality_off"] = ens_q["models"]["tames"]["threshold"]["t_star"]
        for s in seeds:
            w = {k: float(v) for k, v in ens_q["per_seed"][seeds.index(s)]["weights"].items()}
            L = joblib.load(run_dir(cfg) / stage_name("learners", "tames", s, False) / "learners.joblib")
            cnn_q = load_model(cfg, 2, run_dir(cfg) / stage_name("cnn", "tames", s, False) / "model.pt", device,
                               tab["test"]["tab"].shape[1]).eval()
            qoff.append((w, L, cnn_q))
        log("[robustness] ablation-(c) quality-off models found: scoring them as tames_quality_off")
    masks = [load_mask(p) for p in test["mask_path"]]
    rows = []
    conditions = [("clean", 0.0)] + [(k, lv) for k, lvs in DEGRADATIONS.items() for lv in lvs]
    for kind, level in conditions:
        jobs = [(p, mk, kind, level, i, fz, snr0, size, margin, VARIANTS["tames"])
                for i, (p, mk) in enumerate(zip(test["image_path"], masks))]
        with Pool(workers) as pool:
            res = pool.map(_job, jobs, chunksize=8)
        X2 = [r[0] for r in res]; X1 = [r[0][:1] for r in res]; S = [r[1] for r in res]; Q = [r[2] for r in res]
        X2, X1, S, Q = np.stack(X2), np.stack(X1), np.stack(S), np.array(Q, dtype=np.float32)
        s_tilde, obs = tab["pre"].transform(S)
        q_struct = quality_from_indicators(obs)
        tab_in = np.concatenate([q_struct[:, None] * s_tilde, Q[:, None], q_struct[:, None]], axis=1)
        p_t, p_i, p_lf = [], [], []
        for w, L, cnn_t, cnn_i in pieces:
            p_dl, h = predict(cnn_t, X2, device, tab_in)
            z = fuse(h, s_tilde, Q, q_struct)
            probs = {"dl": p_dl, "rf": L["rf"].predict_proba(z)[:, 1], "xgb": L["xgb"].predict_proba(z)[:, 1],
                     "fsvm": L["fsvm"].predict_proba(z)}
            p_t.append(final_probability(ensemble_probability(probs, w), 0.0))
            pi, _ = predict(cnn_i, X1, device)
            p_i.append(pi)
            p_lf.append(0.5 * (pi + L["xgb_struct"].predict_proba(z[:, L["d_h"]:])[:, 1]))
        p_q = []
        if qoff:
            ones = np.ones_like(Q)
            tab_q = np.concatenate([s_tilde, ones[:, None], ones[:, None]], axis=1)   # Q_m = 1
            for w, L, cnn_q in qoff:
                p_dl, h = predict(cnn_q, X2, device, tab_q)
                z = fuse(h, s_tilde, Q, q_struct, quality_on=False)
                probs = {"dl": p_dl, "rf": L["rf"].predict_proba(z)[:, 1], "xgb": L["xgb"].predict_proba(z)[:, 1],
                         "fsvm": L["fsvm"].predict_proba(z)}
                p_q.append(final_probability(ensemble_probability(probs, w), 0.0))
        models_out = [("tames", p_t), ("cnn_image_only", p_i), ("late_fusion_cnn_xgb", p_lf)]
        if p_q:
            models_out.append(("tames_quality_off", p_q))
        for name, P in models_out:
            p = np.mean(P, axis=0)
            met = classification_metrics(y, p, t_star[name])
            ci = bootstrap_ci(y, p, t_star[name], n_boot=n_boot, keys=("auc",))
            rows.append({"degradation": kind, "level": level, "model": name, "auc": met["auc"],
                         "auc_ci_low": ci["auc"][0], "auc_ci_high": ci["auc"][1],
                         "sensitivity": met["sensitivity"], "specificity": met["specificity"],
                         "mean_q": float(Q.mean())})
        log(f"{kind:15s} {level:5.2f}  q̄={Q.mean():.3f}  " + "  ".join(
            f"{r['model']} AUC {r['auc']:.3f}" for r in rows[-len(models_out):]))
    df = pd.DataFrame(rows)
    out = run_dir(cfg) / out_name
    out.mkdir(exist_ok=True)
    df.to_csv(out / "robustness.csv", index=False)
    clean = df[df.degradation == "clean"].set_index("model")["auc"]
    df["auc_drop"] = df.apply(lambda r: clean[r["model"]] - r["auc"], axis=1)
    summ = df[df.degradation != "clean"].groupby("model")["auc_drop"].mean().rename("mean_auc_drop")
    summ.to_csv(out / "robustness_summary.csv")
    log("mean AUC drop over all degradations:\n" + summ.to_string())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        kinds = list(DEGRADATIONS)
        fig, axes = plt.subplots(1, len(kinds), figsize=(4 * len(kinds), 3.4), sharey=True)
        for ax, kind in zip(axes, kinds):
            for name in df.model.unique():
                d = df[(df.model == name) & (df.degradation.isin(["clean", kind]))].sort_values("level")
                ax.plot(range(len(d)), d["auc"], marker="o", label=name)
                ax.fill_between(range(len(d)), d["auc_ci_low"], d["auc_ci_high"], alpha=0.15)
            ax.set_xticks(range(len(DEGRADATIONS[kind]) + 1))
            ax.set_xticklabels(["clean"] + [str(v) for v in DEGRADATIONS[kind]])
            ax.set_title(kind); ax.set_xlabel("severity")
        axes[0].set_ylabel("test AUC"); axes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "robustness.png", dpi=150); plt.close(fig)
    except Exception as e:  # pragma: no cover
        log(f"plot skipped: {e}")
    return df
