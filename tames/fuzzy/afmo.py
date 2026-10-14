"""Adaptive fuzzy morphological operations (AFMO) — Eqs. (5)–(8), plus Dice.

The structuring element is a disc whose radius varies per pixel (Eq. 5). The
reliability-weighted erosion/dilation (Eq. 7) is a convex combination of the
classical flat gray-scale operator and the identity, so both map [0, 1] → [0, 1].
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from skimage.filters import threshold_otsu
from skimage.morphology import disk

EPS = 1e-6


def structuring_radius(sigma: np.ndarray, sigma_g: float, r0: float, gamma: float,
                       r_min: int, r_max: int) -> np.ndarray:
    """Eq. (5): r(p) = clip(round(r0 (1 + γ σ(p)/σ_g)), r_min, r_max)."""
    r = np.round(r0 * (1.0 + gamma * sigma / (sigma_g + EPS)))
    return np.clip(r, r_min, r_max).astype(np.int16)


def reliability_weight(sigma: np.ndarray) -> np.ndarray:
    """Eq. (6): w(p) = 1 − σ(p)/max σ ∈ [0, 1]."""
    return (1.0 - sigma / (float(sigma.max()) + EPS)).astype(np.float32)


def _per_radius(f: np.ndarray, r_map: np.ndarray, r_min: int, r_max: int, op) -> np.ndarray:
    """Apply `op` (min or max filter) with a disc of every radius in [r_min, r_max] and
    pick, per pixel, the result for that pixel's own radius r(p)."""
    out = np.empty_like(f)
    for r in range(r_min, r_max + 1):
        sel = r_map == r
        if not sel.any():
            continue
        out[sel] = op(f, footprint=disk(r), mode="reflect")[sel]
    return out


def weighted_erosion(f: np.ndarray, w: np.ndarray, r_map: np.ndarray, r_min: int, r_max: int) -> np.ndarray:
    """Eq. (7): ε_w(f)(p) = w(p) · min_{B_r(p)} f + (1 − w(p)) · f(p)."""
    m = _per_radius(f, r_map, r_min, r_max, minimum_filter)
    return (w * m + (1.0 - w) * f).astype(np.float32)


def weighted_dilation(f: np.ndarray, w: np.ndarray, r_map: np.ndarray, r_min: int, r_max: int) -> np.ndarray:
    """Eq. (7): δ_w(f)(p) = w(p) · max_{B_r(p)} f + (1 − w(p)) · f(p)."""
    m = _per_radius(f, r_map, r_min, r_max, maximum_filter)
    return (w * m + (1.0 - w) * f).astype(np.float32)


def afmo(mu: np.ndarray, sigma: np.ndarray, sigma_g: float, r0: float, gamma: float,
         r_min: int, r_max: int, lam: float) -> dict[str, np.ndarray]:
    """Eq. (8): O = δ_w(ε_w(μ)), C = ε_w(δ_w(μ)), F = λO + (1−λ)C."""
    r_map = structuring_radius(sigma, sigma_g, r0, gamma, r_min, r_max)
    w = reliability_weight(sigma)
    O = weighted_dilation(weighted_erosion(mu, w, r_map, r_min, r_max), w, r_map, r_min, r_max)
    C = weighted_erosion(weighted_dilation(mu, w, r_map, r_min, r_max), w, r_map, r_min, r_max)
    F = (lam * O + (1.0 - lam) * C).astype(np.float32)
    return {"O": O, "C": C, "F": F, "w": w, "r": r_map}


def otsu_binarize(F: np.ndarray, polarity: str = "low") -> np.ndarray:
    """Binarize a map with Otsu's threshold. `polarity='low'` keeps pixels below the
    threshold (hypoechoic nodules are dark and have low membership at their speckled
    interior); `'high'` keeps pixels above. The polarity is chosen on the validation
    partition by Dice and then fixed."""
    if float(F.max()) - float(F.min()) < EPS:
        return np.zeros(F.shape, dtype=bool)
    t = threshold_otsu(F)
    return (F < t) if polarity == "low" else (F >= t)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice similarity coefficient of two binary masks."""
    a = a.astype(bool)
    b = b.astype(bool)
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / s)
