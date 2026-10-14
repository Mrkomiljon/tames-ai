"""Classification and calibration metrics (Section 3.12)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, matthews_corrcoef, roc_auc_score

EPS = 1e-7


def classification_metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    yhat = (p >= t).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    out = {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
        "accuracy": (tp + tn) / max(len(y), 1), "f1": f1,
        "mcc": float(matthews_corrcoef(y, yhat)) if len(np.unique(yhat)) > 1 else 0.0,
        "threshold": float(t), "n": int(len(y)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    return out


def calibration_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Brier score, calibration slope/intercept (logistic recalibration) and a
    quantile-binned reliability table."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    lg = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6).fit(lg, y)          # calibration slope
    a = _intercept_given_slope_one(lg.ravel(), y)          # calibration-in-the-large
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    bins = []
    for i in range(n_bins):
        sel = (p >= edges[i]) & (p <= edges[i + 1] if i == n_bins - 1 else p < edges[i + 1])
        if sel.sum() == 0:
            continue
        bins.append({"bin": i, "n": int(sel.sum()), "mean_pred": float(p[sel].mean()),
                     "obs_rate": float(y[sel].mean())})
    return {"brier": float(brier_score_loss(y, p)), "slope": float(slope.coef_[0, 0]),
            "intercept": a, "bins": bins}


def _intercept_given_slope_one(lg: np.ndarray, y: np.ndarray) -> float:
    """Calibration-in-the-large: a maximizing Σ y log σ(a+lg) + (1−y) log(1−σ(a+lg))."""
    a = 0.0
    for _ in range(100):
        s = 1 / (1 + np.exp(-(a + lg)))
        g = float((y - s).sum())
        h = float((s * (1 - s)).sum())
        if h < EPS:
            break
        step = g / h
        a += step
        if abs(step) < 1e-8:
            break
    return float(a)
