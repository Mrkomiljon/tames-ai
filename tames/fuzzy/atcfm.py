"""Adaptive texture–contrast fuzzy membership (ATCFM) — Eqs. (1)–(4).

All maps are float32 arrays with values in [0, 1] and the same H×W shape as the
input image, except the scalars returned by `immerkaer_noise`, `snr` and
`quality_coefficient`.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, maximum_filter, minimum_filter, uniform_filter
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import footprint_rectangle

EPS = 1e-6
L_BINS = 256  # gray levels; log2(L) = 8 normalizes the local entropy


def normalize(img: np.ndarray) -> np.ndarray:
    """Eq. (1): min–max normalization to [0, 1]."""
    x = img.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < EPS:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def local_stats(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (2): local mean c(p) and local standard deviation σ(p) over a k×k window."""
    c = uniform_filter(x, size=k, mode="reflect")
    m2 = uniform_filter(x * x, size=k, mode="reflect")
    var = np.clip(m2 - c * c, 0.0, None)
    return c.astype(np.float32), np.sqrt(var).astype(np.float32)


def local_entropy(x: np.ndarray, k: int) -> np.ndarray:
    """Eq. (3), T(p): Shannon entropy of the L-bin histogram of W_p divided by log2 L."""
    x8 = np.clip(np.round(x * (L_BINS - 1)), 0, L_BINS - 1).astype(np.uint8)
    h = rank_entropy(x8, footprint_rectangle((k, k)))  # bits, 8-bit image → max 8
    return np.clip(h / np.log2(L_BINS), 0.0, 1.0).astype(np.float32)


def local_contrast(x: np.ndarray, k: int) -> np.ndarray:
    """Eq. (3), K(p): Michelson contrast of W_p."""
    mx = maximum_filter(x, size=k, mode="reflect")
    mn = minimum_filter(x, size=k, mode="reflect")
    return ((mx - mn) / (mx + mn + EPS)).astype(np.float32)


_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def immerkaer_noise(x: np.ndarray) -> float:
    """Noise standard deviation σ_noise by the Laplacian method of Immerkær (1996)."""
    h, w = x.shape
    conv = convolve(x.astype(np.float32), _IMMERKAER, mode="reflect")
    s = float(np.abs(conv).sum())
    return float(np.sqrt(np.pi / 2.0) / (6.0 * (w - 2) * (h - 2)) * s)


def snr(x: np.ndarray) -> float:
    """Eq. (3): SNR = mean(x̃) / σ_noise."""
    return float(x.mean()) / (immerkaer_noise(x) + EPS)


def quality_coefficient(snr_value: float, snr0: float) -> float:
    """Eq. (3), q = SNR / (SNR + SNR_0) ∈ [0, 1]; SNR_0 is the median SNR of the training images."""
    return float(snr_value / (snr_value + snr0 + EPS))


def atcfm(x: np.ndarray, c: np.ndarray, sigma: np.ndarray, T: np.ndarray, K: np.ndarray,
          q: float, alpha: float, beta: float, eps: float = EPS) -> np.ndarray:
    """Eq. (4): adaptive membership μ(p) ∈ [0, 1].

    μ = q · exp(−(x̃−c)² / (2σ²+ε)) · (1+αT)(1+βK) / ((1+α)(1+β)).
    With α = β = 0 and q = 1 this is the plain local Gaussian membership.
    """
    g = np.exp(-((x - c) ** 2) / (2.0 * sigma ** 2 + eps))
    mod = (1.0 + alpha * T) * (1.0 + beta * K) / ((1.0 + alpha) * (1.0 + beta))
    mu = q * g * mod
    return np.clip(mu, 0.0, 1.0).astype(np.float32)
