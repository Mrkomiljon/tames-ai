"""Adaptive ensemble, regional prior and decision rule — Eqs. (14)–(16)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_curve

EPS = 1e-7


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def adaptive_weights(val_losses: dict[str, float], tau: float) -> dict[str, float]:
    """Eq. (14): ω_k = exp(−L_k^val/τ) / Σ_j exp(−L_j^val/τ)."""
    names = list(val_losses)
    L = np.array([val_losses[k] for k in names], dtype=np.float64)
    a = -L / tau
    a -= a.max()
    w = np.exp(a)
    w /= w.sum()
    return dict(zip(names, w.astype(float)))


def ensemble_probability(probs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Eq. (14): p_ens = Σ_k ω_k p_k."""
    return sum(weights[k] * np.asarray(probs[k], dtype=np.float64) for k in weights)


def select_tau(val_probs: dict[str, np.ndarray], y_va: np.ndarray, grid: list[float]
               ) -> tuple[float, dict[str, float], dict[str, float]]:
    """Choose τ on the validation partition by ensemble log-loss (Section 3.8)."""
    losses = {k: float(log_loss(y_va, np.clip(p, EPS, 1 - EPS))) for k, p in val_probs.items()}
    best = (np.inf, None, None)
    for tau in grid:
        w = adaptive_weights(losses, float(tau))
        ll = log_loss(y_va, np.clip(ensemble_probability(val_probs, w), EPS, 1 - EPS))
        if ll < best[0]:
            best = (ll, float(tau), w)
    return best[1], best[2], losses


def regional_prior_shift(pi_reg: float, pi_train: float) -> float:
    """Eq. (15): Δ_reg = logit π_reg − logit π_train (0 when the regions coincide)."""
    return float(logit(np.array([pi_reg]))[0] - logit(np.array([pi_train]))[0])


def final_probability(p_ens: np.ndarray, delta_reg: float = 0.0) -> np.ndarray:
    """Eq. (16): p_final = σ(logit p_ens + Δ_reg)."""
    return sigmoid(logit(p_ens) + delta_reg)


def choose_threshold(y: np.ndarray, p: np.ndarray, min_sensitivity: float = 0.95) -> dict[str, float]:
    """Eq. (16): t* fixed on validation — sensitivity ≥ `min_sensitivity` with maximal
    specificity; the Youden-optimal threshold is returned alongside."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = tpr >= min_sensitivity
    if ok.any():
        i = np.argmax(np.where(ok, 1.0 - fpr, -np.inf))
        t_sens = float(thr[i])
    else:
        t_sens = float(thr[np.argmax(tpr)])
    j = np.argmax(tpr - fpr)
    t_sens = float(np.clip(t_sens, 0.0, 1.0))
    return {"t_star": t_sens, "t_youden": float(np.clip(thr[j], 0.0, 1.0))}
