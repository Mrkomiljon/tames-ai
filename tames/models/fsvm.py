"""Fuzzy support vector machine (Lin & Wang, 2002) — Eqs. (12)–(13).

Eq. (12) is the standard soft-margin SVM with a per-sample penalty C·s_n; in
scikit-learn that is `SVC(C=C).fit(z, y, sample_weight=s)`. Eq. (13) gives the
membership s_n ∈ (0, 1] from the distance to the class centroid in the
standardized feature space. The decision value is mapped to a probability by
Platt scaling (Platt, 1999) fitted on the validation partition.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def sample_membership(z: np.ndarray, y: np.ndarray, delta: float = 1e-3) -> np.ndarray:
    """Eq. (13): s_n = 1 − ||z_n − z̄_y|| / (max_{m: y_m = y_n} ||z_m − z̄_y|| + δ)."""
    s = np.empty(len(y), dtype=np.float32)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        centroid = z[idx].mean(axis=0)
        d = np.linalg.norm(z[idx] - centroid, axis=1)
        s[idx] = 1.0 - d / (d.max() + delta)
    return np.clip(s, 1e-6, 1.0)


class FuzzySVM:
    def __init__(self, C: float = 1.0, gamma: str | float = "scale", delta: float = 1e-3, seed: int = 0):
        self.scaler = StandardScaler()
        self.svc = SVC(C=C, kernel="rbf", gamma=gamma, random_state=seed)
        self.platt = LogisticRegression(C=1e6)
        self.delta = delta
        self.params = {"C": C, "gamma": gamma}

    def fit(self, z_tr: np.ndarray, y_tr: np.ndarray) -> "FuzzySVM":
        zs = self.scaler.fit_transform(z_tr)
        s = sample_membership(zs, y_tr, self.delta)
        self.svc.fit(zs, y_tr, sample_weight=s)
        self.s_ = s
        return self

    def decision(self, z: np.ndarray) -> np.ndarray:
        return self.svc.decision_function(self.scaler.transform(z))

    def fit_platt(self, z_va: np.ndarray, y_va: np.ndarray) -> "FuzzySVM":
        d = self.decision(z_va).reshape(-1, 1)
        self.platt.fit(d, y_va)
        return self

    def predict_proba(self, z: np.ndarray) -> np.ndarray:
        d = self.decision(z).reshape(-1, 1)
        return self.platt.predict_proba(d)[:, 1]


def fit_fsvm(z_tr, y_tr, z_va, y_va, grid: dict, seed: int = 0):
    """Select C and γ on validation AUC of the decision value, then fit Platt scaling."""
    best = (-1.0, None)
    for C, g in itertools.product(grid["C"], grid["gamma"]):
        m = FuzzySVM(C=float(C), gamma=g, seed=seed).fit(z_tr, y_tr)
        auc = roc_auc_score(y_va, m.decision(z_va))
        if auc > best[0]:
            best = (auc, m)
    best[1].fit_platt(z_va, y_va)
    return best[1], best[1].params, best[0]
