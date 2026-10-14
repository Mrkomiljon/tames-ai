"""The three stages after preprocessing, as importable functions used by scripts 03–06.

    cnn_stage       f_θ + DL learner (Eq. 9, Section 3.7): out-of-fold embeddings for the
                    training rows, a final model for validation/test rows.
    learners_stage  z (Eqs. 10–11) → RF, XGBoost, Fuzzy SVM (Eqs. 12–13) + structured-only
                    XGBoost baseline.
    ensemble_stage  Eqs. (14)–(16), baselines, metrics, CIs, DeLong, McNemar, calibration.

Channel layout of X_{split}.npy (from scripts/02_fuzzy_preprocess.py): 0 = x̃, 1 = μ,
2 = F, 3 = F_plain. Variants pick channels:
    tames           [x̃, F]          full model
    image_only      [x̃]             image-only CNN baseline (no AFMO channel, no tabular input)
    abl_afmo_off    [x̃, μ]          ablation (a): F replaced by μ
    abl_atcfm_off   [x̃, F_plain]    ablation (b): plain Gaussian membership (α = β = 0, q = 1)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .config import run_dir
from .eval import (bootstrap_ci, bootstrap_metric_difference, calibration_metrics,
                   classification_metrics, delong_test, mcnemar_exact)
from .eval.stats import holm
from .features import DESCRIPTOR_NAMES, TabularPreprocessor, fuse, quality_from_indicators
from .models import (choose_threshold, ensemble_probability, final_probability, get_device,
                     regional_prior_shift)
from .models.cnn import load_model, predict, save_model, train_cnn
from .models.ensemble import select_tau
from .models.fsvm import fit_fsvm
from .models.learners import fit_rf, fit_xgb, rf_train_probability

VARIANTS = {"tames": [0, 2], "image_only": [0], "abl_afmo_off": [0, 1], "abl_atcfm_off": [0, 3]}
SPLITS = ("train", "val", "test")
LEARNERS = ("dl", "rf", "xgb", "fsvm")


# ------------------------------------------------------------------ loading ---
def load_processed(cfg: dict) -> dict:
    """Load the step-2 outputs. With `train_frac` < 1 in the config (low-data experiment)
    the training partition is subsampled once, stratified by label and seeded by
    `cfg['seed']`; validation and test are never touched."""
    proc = Path(cfg["paths"]["processed"])
    out = {"meta": {}, "X": {}, "y": {}}
    for s in SPLITS:
        out["meta"][s] = pd.read_csv(proc / f"meta_{s}.csv")
        out["X"][s] = np.load(proc / f"X_{s}.npy", mmap_mode="r")
        out["y"][s] = out["meta"][s]["label"].values.astype(int)
    frac = float(cfg.get("train_frac", 1.0) or 1.0)
    if frac < 1.0:
        y = out["y"]["train"]
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        keep = np.concatenate([rng.choice(np.where(y == c)[0], size=max(2, int(round(frac * (y == c).sum()))),
                                          replace=False) for c in (0, 1)])
        keep.sort()
        out["meta"]["train"] = out["meta"]["train"].iloc[keep].reset_index(drop=True)
        out["X"]["train"] = np.asarray(out["X"]["train"])[keep]
        out["y"]["train"] = y[keep]
        out["train_frac"] = frac
    return out


def tabular_block(data: dict, quality_on: bool = True) -> dict:
    """s̃, observation indicators, Q_struct and q for every split (fit on train only)."""
    pre = TabularPreprocessor().fit(data["meta"]["train"][DESCRIPTOR_NAMES].values)
    out = {"pre": pre, "names": list(DESCRIPTOR_NAMES)}
    for s in SPLITS:
        S = data["meta"][s][DESCRIPTOR_NAMES].values
        s_tilde, obs = pre.transform(S)
        q_struct = quality_from_indicators(obs)
        q_img = data["meta"][s]["q"].values.astype(np.float32)
        if not quality_on:                      # ablation (c)
            q_struct = np.ones_like(q_struct)
            q_img = np.ones_like(q_img)
        out[s] = {"s_tilde": s_tilde, "q_struct": q_struct, "q_img": q_img,
                  "tab": np.concatenate([q_struct[:, None] * s_tilde, q_img[:, None], q_struct[:, None]], axis=1)}
    return out


def stage_name(kind: str, variant: str, seed: int, quality_on: bool = True) -> str:
    return f"{kind}_{variant}{'' if quality_on else '_qoff'}_seed{seed}"


# ---------------------------------------------------------------- CNN stage ---
def cnn_stage(cfg: dict, variant: str, seed: int = 0, quality_on: bool = True, log=print) -> Path:
    data = load_processed(cfg)
    chans = VARIANTS[variant]
    use_tab = variant != "image_only"
    tab = tabular_block(data, quality_on) if use_tab else None
    device = get_device(cfg.get("device", "auto"))
    out = run_dir(cfg) / stage_name("cnn", variant, seed, quality_on)
    out.mkdir(parents=True, exist_ok=True)
    X = {s: np.asarray(data["X"][s][:, chans], dtype=np.float32) for s in SPLITS}
    y = data["y"]
    T = {s: (tab[s]["tab"] if use_tab else None) for s in SPLITS}
    log(f"[cnn] variant={variant} channels={chans} tab={'yes' if use_tab else 'no'} device={device} "
        f"n train/val/test = {len(y['train'])}/{len(y['val'])}/{len(y['test'])}")

    # One model trained on the full training partition, early-stopped on validation.
    # Its embedding space is shared by every row (train rows are in-sample; the
    # downstream learners are therefore tuned on validation only — Section 3.6).
    model, hist = train_cnn(cfg, X["train"], y["train"], X["val"], y["val"], len(chans), device,
                            T["train"], T["val"], seed=seed, log=log)
    save_model(model, out / "model.pt")
    for s in SPLITS:
        p, e = predict(model, X[s], device, T[s])
        np.save(out / f"p_{s}.npy", p)
        np.save(out / f"emb_{s}.npy", e)
    res = {"variant": variant, "channels": chans, "seed": seed, "quality_on": quality_on,
           "train_auc_in_sample": float(roc_auc_score(y["train"], np.load(out / "p_train.npy"))),
           "val_auc": float(roc_auc_score(y["val"], np.load(out / "p_val.npy"))),
           "test_auc": float(roc_auc_score(y["test"], np.load(out / "p_test.npy"))),
           "best_val_auc_during_training": hist["best_val_auc"], "history": hist["history"],
           "d_tab": 0 if not use_tab else int(T["train"].shape[1])}
    json.dump(res, open(out / "cnn_results.json", "w"), indent=2)
    log(f"[cnn] train AUC (in-sample) {res['train_auc_in_sample']:.3f}  val {res['val_auc']:.3f}  "
        f"test {res['test_auc']:.3f}")
    return out


# ----------------------------------------------------------- learners stage ---
def build_z(cfg: dict, variant: str, seed: int, quality_on: bool = True) -> dict:
    data = load_processed(cfg)
    tab = tabular_block(data, quality_on)
    cdir = run_dir(cfg) / stage_name("cnn", variant, seed, quality_on)
    z, names = {}, None
    for s in SPLITS:
        h = np.load(cdir / f"emb_{s}.npy")
        z[s] = fuse(h, tab[s]["s_tilde"], tab[s]["q_img"], tab[s]["q_struct"], quality_on=quality_on)
        if names is None:
            names = [f"h_img_{i}" for i in range(h.shape[1])] + list(tab["names"]) + ["Q_img", "Q_struct"]
    return {"z": z, "y": data["y"], "names": names, "tab": tab, "cnn_dir": cdir}


def learners_stage(cfg: dict, variant: str, seed: int = 0, quality_on: bool = True, log=print) -> Path:
    b = build_z(cfg, variant, seed, quality_on)
    z, y, g = b["z"], b["y"], cfg["learners"]
    out = run_dir(cfg) / stage_name("learners", variant, seed, quality_on)
    out.mkdir(parents=True, exist_ok=True)
    params = {}
    log(f"[learners] variant={variant} d_z={z['train'].shape[1]}")
    rf, params["rf"], auc = fit_rf(z["train"], y["train"], z["val"], y["val"], g["rf"], seed)
    log(f"[learners] RF   val AUC {auc:.3f}  {params['rf']}")
    np.save(out / "p_rf_train.npy", rf_train_probability(rf))
    for s in ("val", "test"):
        np.save(out / f"p_rf_{s}.npy", rf.predict_proba(z[s])[:, 1])
    xgb, params["xgb"], auc = fit_xgb(z["train"], y["train"], z["val"], y["val"], g["xgb"], seed)
    log(f"[learners] XGB  val AUC {auc:.3f}  {params['xgb']}")
    for s in ("val", "test"):
        np.save(out / f"p_xgb_{s}.npy", xgb.predict_proba(z[s])[:, 1])
    fsvm, params["fsvm"], auc = fit_fsvm(z["train"], y["train"], z["val"], y["val"], g["fsvm"], seed)
    log(f"[learners] FSVM val AUC {auc:.3f}  {params['fsvm']}")
    for s in ("val", "test"):
        np.save(out / f"p_fsvm_{s}.npy", fsvm.predict_proba(z[s]))
    # structured-only XGBoost baseline (descriptors only, no image embedding)
    d_h = len([n for n in b["names"] if n.startswith("h_img_")])
    zs = {s: z[s][:, d_h:] for s in SPLITS}
    xgb_s, params["xgb_struct"], auc = fit_xgb(zs["train"], y["train"], zs["val"], y["val"], g["xgb"], seed)
    log(f"[learners] XGB (structured only) val AUC {auc:.3f}")
    for s in ("val", "test"):
        np.save(out / f"p_xgb_struct_{s}.npy", xgb_s.predict_proba(zs[s])[:, 1])
    for s in SPLITS:
        np.save(out / f"z_{s}.npy", z[s])
    import joblib
    joblib.dump({"rf": rf, "xgb": xgb, "fsvm": fsvm, "xgb_struct": xgb_s, "names": b["names"],
                 "d_h": d_h}, out / "learners.joblib")
    json.dump({"variant": variant, "seed": seed, "quality_on": quality_on, "params": params,
               "feature_names": b["names"]}, open(out / "learner_params.json", "w"), indent=2)
    return out


# ----------------------------------------------------------- ensemble stage ---
def _load_probs(cfg: dict, variant: str, seed: int, quality_on: bool) -> dict:
    cdir = run_dir(cfg) / stage_name("cnn", variant, seed, quality_on)
    ldir = run_dir(cfg) / stage_name("learners", variant, seed, quality_on)
    P = {}
    for s in ("val", "test"):
        P[s] = {"dl": np.load(cdir / f"p_{s}.npy"), "rf": np.load(ldir / f"p_rf_{s}.npy"),
                "xgb": np.load(ldir / f"p_xgb_{s}.npy"), "fsvm": np.load(ldir / f"p_fsvm_{s}.npy"),
                "xgb_struct": np.load(ldir / f"p_xgb_struct_{s}.npy")}
        io = run_dir(cfg) / stage_name("cnn", "image_only", seed, True)
        if (io / f"p_{s}.npy").exists():
            P[s]["image_only"] = np.load(io / f"p_{s}.npy")
    return P


def _model_set(P: dict, learners: tuple, tau_grid, y_val, equal_weights: bool) -> dict:
    val_probs = {k: P["val"][k] for k in learners}
    if equal_weights:
        tau, weights, losses = float("inf"), {k: 1.0 / len(learners) for k in learners}, \
            {k: float(log_loss(y_val, np.clip(p, 1e-7, 1 - 1e-7))) for k, p in val_probs.items()}
    else:
        tau, weights, losses = select_tau(val_probs, y_val, tau_grid)
    return {"tau": tau, "weights": weights, "val_losses": losses}


def ensemble_stage(cfg: dict, variant: str = "tames", seeds=(0,), quality_on: bool = True,
                   equal_weights: bool = False, drop: str | None = None, tag: str | None = None,
                   log=print) -> dict:
    """Evaluate the ensemble (and all baselines available) on validation and test.
    Probabilities are averaged over `seeds` before the metrics ('pooled predictions');
    per-seed AUCs give the seed-to-seed SD."""
    data = load_processed(cfg)
    y = data["y"]
    ens = cfg["ensemble"]
    learners = tuple(k for k in LEARNERS if k != drop)
    per_seed, pooled = [], {"val": {}, "test": {}}
    for seed in seeds:
        P = _load_probs(cfg, variant, seed, quality_on)
        ms = _model_set(P, learners, ens["tau_grid"], y["val"], equal_weights)
        pi_train = float(y["train"].mean())
        pi_reg = ens.get("pi_reg")
        delta = 0.0 if pi_reg is None else regional_prior_shift(float(pi_reg), pi_train)
        models = {}
        for s in ("val", "test"):
            p_ens = ensemble_probability(P[s], ms["weights"])
            models[s] = {"tames": final_probability(p_ens, delta),
                         **{f"{k}_only": P[s][k] for k in LEARNERS if k in P[s]},
                         "xgb_struct_only": P[s]["xgb_struct"]}
            if "image_only" in P[s]:
                models[s]["cnn_image_only"] = P[s]["image_only"]
                models[s]["late_fusion_cnn_xgb"] = 0.5 * (P[s]["image_only"] + P[s]["xgb_struct"])
        per_seed.append({"seed": seed, **ms, "delta_reg": delta,
                         "test_auc": {m: float(roc_auc_score(y["test"], p)) for m, p in models["test"].items()}})
        for s in ("val", "test"):
            for m, p in models[s].items():
                pooled[s].setdefault(m, []).append(np.asarray(p, dtype=np.float64))
    for s in ("val", "test"):
        pooled[s] = {m: np.mean(v, axis=0) for m, v in pooled[s].items()}
    # thresholds on validation, metrics with CIs on test
    n_boot = int(cfg["eval"]["n_boot"])
    results = {"variant": variant, "seeds": list(seeds), "quality_on": quality_on, "equal_weights": equal_weights,
               "drop": drop, "learners": learners, "per_seed": per_seed, "models": {}}
    for m, p_val in pooled["val"].items():
        thr = choose_threshold(y["val"], p_val, float(ens["min_sensitivity"]))
        p_te = pooled["test"][m]
        r = {"threshold": thr, "val": classification_metrics(y["val"], p_val, thr["t_star"]),
             "test": classification_metrics(y["test"], p_te, thr["t_star"]),
             "test_youden": classification_metrics(y["test"], p_te, thr["t_youden"]),
             "test_ci": bootstrap_ci(y["test"], p_te, thr["t_star"], n_boot=n_boot, seed=0),
             "calibration_test": calibration_metrics(y["test"], p_te)}
        aucs = [ps["test_auc"][m] for ps in per_seed if m in ps["test_auc"]]
        r["test_auc_seed_mean"], r["test_auc_seed_sd"] = float(np.mean(aucs)), float(np.std(aucs))
        results["models"][m] = r
    # comparisons of the full model with every baseline
    comp, pvals = {}, {}
    yt = y["test"]
    pa, ta = pooled["test"]["tames"], results["models"]["tames"]["threshold"]["t_star"]
    for m in pooled["test"]:
        if m == "tames":
            continue
        pb, tb = pooled["test"][m], results["models"][m]["threshold"]["t_star"]
        d = delong_test(yt, pa, pb)
        comp[m] = {"delong": d,
                   "auc_diff_bootstrap": bootstrap_metric_difference(yt, pa, pb, ta, tb, "auc", n_boot),
                   "mcnemar_sensitivity": mcnemar_exact(yt, (pa >= ta).astype(int), (pb >= tb).astype(int), 1),
                   "mcnemar_specificity": mcnemar_exact(yt, (pa >= ta).astype(int), (pb >= tb).astype(int), 0)}
        pvals[m] = d["p"]
    adj = holm(pvals)
    for m in comp:
        comp[m]["delong"]["p_holm"] = adj[m]
    results["comparisons_vs_tames"] = comp
    out = run_dir(cfg) / (tag or stage_name("ensemble", variant, 0, quality_on)
                          + ("_equal" if equal_weights else "") + (f"_drop_{drop}" if drop else ""))
    out.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out / "results.json", "w"), indent=2, default=float)
    for s in ("val", "test"):
        np.save(out / f"p_pooled_{s}.npy", np.stack([pooled[s][m] for m in results["models"]]))
    json.dump(list(results["models"]), open(out / "model_order.json", "w"))
    results_table(results).to_csv(out / "results_table.csv", index=False)
    log(results_table(results).to_string(index=False))
    log(f"[ensemble] weights {per_seed[0]['weights']}  tau {per_seed[0]['tau']}  Δ_reg {per_seed[0]['delta_reg']:.3f}")
    return results


def results_table(results: dict) -> pd.DataFrame:
    rows = []
    for m, r in results["models"].items():
        t, ci = r["test"], r["test_ci"]
        rows.append({"model": m, "AUC": f"{t['auc']:.3f} [{ci['auc'][0]:.3f}, {ci['auc'][1]:.3f}]",
                     "Sens%": f"{100 * t['sensitivity']:.1f} [{100 * ci['sensitivity'][0]:.1f}, {100 * ci['sensitivity'][1]:.1f}]",
                     "Spec%": f"{100 * t['specificity']:.1f} [{100 * ci['specificity'][0]:.1f}, {100 * ci['specificity'][1]:.1f}]",
                     "Acc%": f"{100 * t['accuracy']:.1f}", "F1%": f"{100 * t['f1']:.1f}", "MCC": f"{t['mcc']:.3f}",
                     "Brier": f"{r['calibration_test']['brier']:.3f}", "t*": f"{t['threshold']:.3f}",
                     "AUC seed SD": f"{r['test_auc_seed_sd']:.3f}"})
    return pd.DataFrame(rows)
