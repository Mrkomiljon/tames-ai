"""Quality-weighted multimodal fusion — Eqs. (10)–(11).

z = [ Q_img · h_img ; Q_struct · s̃ ; Q_img, Q_struct ]
where h_img is the CNN embedding, s̃ the z-scored (training statistics) and
imputed structured descriptors, Q_img = q (Eq. 3) and Q_struct the observed
fraction of descriptor values (Eq. 10). The two coefficients are appended as
features so that tree learners can model quality–value interactions.
"""
from __future__ import annotations

import numpy as np


def quality_from_indicators(observed: np.ndarray) -> np.ndarray:
    """Eq. (10): Q_m = (1/d_m) Σ_j o_mj for each row; `observed` is (n, d_m) in {0,1}."""
    return observed.mean(axis=1).astype(np.float32)


class TabularPreprocessor:
    """Median imputation + z-scoring with training-partition statistics, and the
    observation indicators needed by Eq. (10). Fit on the training rows only."""

    def __init__(self):
        self.median_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, S: np.ndarray) -> "TabularPreprocessor":
        S = np.asarray(S, dtype=np.float64)
        self.median_ = np.nanmedian(S, axis=0)
        self.median_ = np.where(np.isnan(self.median_), 0.0, self.median_)
        imp = np.where(np.isnan(S), self.median_, S)
        self.mean_ = imp.mean(axis=0)
        self.std_ = imp.std(axis=0) + 1e-6
        return self

    def transform(self, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (s̃, observed) with s̃ (n, d) z-scored and observed (n, d) in {0,1}."""
        S = np.asarray(S, dtype=np.float64)
        observed = (~np.isnan(S)).astype(np.float32)
        imp = np.where(np.isnan(S), self.median_, S)
        return ((imp - self.mean_) / self.std_).astype(np.float32), observed


def fuse(h_img: np.ndarray, s_tilde: np.ndarray, q_img: np.ndarray, q_struct: np.ndarray,
         *, quality_on: bool = True) -> np.ndarray:
    """Eq. (11). `quality_on=False` is ablation (c): every Q_m = 1."""
    q_img = np.asarray(q_img, dtype=np.float32).reshape(-1, 1)
    q_struct = np.asarray(q_struct, dtype=np.float32).reshape(-1, 1)
    if not quality_on:
        q_img = np.ones_like(q_img)
        q_struct = np.ones_like(q_struct)
    return np.concatenate([q_img * h_img, q_struct * s_tilde, q_img, q_struct], axis=1).astype(np.float32)
