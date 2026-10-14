"""Statistical analysis (Section 3.12): bootstrap CIs, DeLong test, exact McNemar test."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .metrics import classification_metrics


def bootstrap_ci(y: np.ndarray, p: np.ndarray, t: float, n_boot: int = 2000, seed: int = 0,
                 keys=("auc", "sensitivity", "specificity", "accuracy", "f1", "mcc", "ppv", "npv")
                 ) -> dict[str, tuple[float, float]]:
    """Percentile 95 % intervals from `n_boot` resamples of the evaluation rows."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        m = classification_metrics(y[idx], p[idx], t)
        for k in keys:
            acc[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in acc.items()}


def bootstrap_metric_difference(y, p_a, p_b, t_a, t_b, metric: str = "auc", n_boot: int = 2000,
                                seed: int = 0) -> dict[str, float]:
    """Paired bootstrap of metric(A) − metric(B) on the same rows."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        ma = classification_metrics(y[idx], p_a[idx], t_a)[metric]
        mb = classification_metrics(y[idx], p_b[idx], t_b)[metric]
        diffs.append(ma - mb)
    diffs = np.array(diffs)
    return {"diff": float(classification_metrics(y, p_a, t_a)[metric] - classification_metrics(y, p_b, t_b)[metric]),
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5))}


# ---------------------------------------------------------------- DeLong (1988) ---
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T + 1
    return T2


def _fast_delong(preds_sorted_transposed: np.ndarray, m: int):
    """Sun & Xu (2014) fast DeLong. `preds_sorted_transposed` is (k, n) with the m
    positive examples first. Returns AUCs (k,) and the covariance (k, k)."""
    n = preds_sorted_transposed.shape[1] - m
    k = preds_sorted_transposed.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(preds_sorted_transposed[r, :m])
        ty[r] = _compute_midrank(preds_sorted_transposed[r, m:])
        tz[r] = _compute_midrank(preds_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    return aucs, sx / m + sy / n


def delong_test(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> dict[str, float]:
    """Two-sided DeLong test for the difference of two correlated AUCs."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y)
    m = int(y.sum())
    preds = np.vstack([np.asarray(p_a)[order], np.asarray(p_b)[order]])
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(max(var, 1e-12))
    pval = 2 * stats.norm.sf(abs(z))
    se = float(np.sqrt(max(var, 1e-12)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": float(diff),
            "ci_low": float(diff - 1.96 * se), "ci_high": float(diff + 1.96 * se),
            "z": float(z), "p": float(pval)}


def mcnemar_exact(y: np.ndarray, yhat_a: np.ndarray, yhat_b: np.ndarray, positive_class: int
                  ) -> dict[str, float]:
    """Exact McNemar test on the discordant pairs within one class: sensitivity
    (positive_class=1) or specificity (positive_class=0)."""
    y, a, b = np.asarray(y), np.asarray(yhat_a), np.asarray(yhat_b)
    sel = y == positive_class
    ca = a[sel] == positive_class
    cb = b[sel] == positive_class
    n_ab = int((ca & ~cb).sum())   # A right, B wrong
    n_ba = int((~ca & cb).sum())   # B right, A wrong
    n_disc = n_ab + n_ba
    pval = float(stats.binomtest(n_ab, n_disc, 0.5).pvalue) if n_disc > 0 else 1.0
    return {"b_A_right_B_wrong": n_ab, "c_B_right_A_wrong": n_ba, "p": pval}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down correction."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out
