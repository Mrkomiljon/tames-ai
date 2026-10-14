"""Range and reduction checks for Eqs. (1)–(8), (13), (14).

Run:  python -m pytest tests -q
"""
import numpy as np

from tames.fuzzy import (afmo, atcfm, dice, local_contrast, local_entropy, local_stats,
                         normalize, otsu_binarize, quality_coefficient, reliability_weight,
                         snr, structuring_radius, weighted_dilation, weighted_erosion)
from tames.models.ensemble import adaptive_weights
from tames.models.fsvm import sample_membership

rng = np.random.default_rng(0)
IMG = (rng.random((96, 128)) * 255).astype(np.uint8)
IMG[30:60, 40:90] = (IMG[30:60, 40:90] * 0.3).astype(np.uint8)  # a dark "nodule"


def _maps(alpha=1.0, beta=0.5, q=0.8, k=7):
    x = normalize(IMG)
    c, s = local_stats(x, k)
    T = local_entropy(x, k)
    K = local_contrast(x, k)
    return x, c, s, T, K, atcfm(x, c, s, T, K, q, alpha, beta)


def test_normalize_range():
    x = normalize(IMG)
    assert x.min() == 0.0 and x.max() == 1.0 and x.dtype == np.float32


def test_descriptors_in_unit_interval():
    x, c, s, T, K, mu = _maps()
    for name, m in {"c": c, "sigma": s, "T": T, "K": K, "mu": mu}.items():
        assert 0.0 <= m.min() and m.max() <= 1.0, name


def test_mu_bounded_by_q():
    q = 0.37
    *_, mu = _maps(q=q)
    assert mu.max() <= q + 1e-6


def test_mu_reduces_to_gaussian():
    x, c, s, T, K, mu = _maps(alpha=0.0, beta=0.0, q=1.0)
    g = np.exp(-((x - c) ** 2) / (2 * s ** 2 + 1e-6))
    assert np.allclose(mu, g, atol=1e-6)


def test_quality_coefficient():
    x = normalize(IMG)
    s = snr(x)
    assert s > 0
    q = quality_coefficient(s, s)
    assert abs(q - 0.5) < 1e-6
    assert 0 < quality_coefficient(s, 10 * s) < 0.5


def test_radius_and_weight():
    x, c, s, *_ = _maps()
    r = structuring_radius(s, float(x.std()), r0=2, gamma=1.0, r_min=1, r_max=5)
    assert r.min() >= 1 and r.max() <= 5
    w = reliability_weight(s)
    assert 0.0 <= w.min() and w.max() <= 1.0
    assert np.isclose(w.min(), 0.0, atol=1e-4)  # the noisiest pixel gets weight ≈ 0


def test_operators_map_unit_interval_and_limits():
    x, c, s, T, K, mu = _maps()
    r = structuring_radius(s, float(x.std()), 2, 1.0, 1, 3)
    w = reliability_weight(s)
    e = weighted_erosion(mu, w, r, 1, 3)
    d = weighted_dilation(mu, w, r, 1, 3)
    assert 0 <= e.min() and e.max() <= 1 and 0 <= d.min() and d.max() <= 1
    assert np.all(e <= mu + 1e-6) and np.all(d >= mu - 1e-6)
    # w = 0 → identity; w = 1 → classical operators (erosion ≤ identity ≤ dilation)
    assert np.allclose(weighted_erosion(mu, np.zeros_like(w), r, 1, 3), mu)
    assert np.allclose(weighted_dilation(mu, np.zeros_like(w), r, 1, 3), mu)
    e1 = weighted_erosion(mu, np.ones_like(w), r, 1, 3)
    assert np.all(e1 <= e + 1e-6)


def test_afmo_convex():
    x, c, s, T, K, mu = _maps()
    out = afmo(mu, s, float(x.std()), 2, 1.0, 1, 3, lam=0.4)
    O, C, F = out["O"], out["C"], out["F"]
    assert np.allclose(F, 0.4 * O + 0.6 * C)
    assert 0 <= F.min() and F.max() <= 1


def test_dice_and_otsu():
    a = np.zeros((10, 10), bool); a[2:6, 2:6] = True
    assert dice(a, a) == 1.0 and dice(a, ~a) == 0.0
    x, *_, mu = _maps()
    m = otsu_binarize(mu, "low")
    assert m.dtype == bool and m.shape == mu.shape


def test_sample_membership_range():
    z = rng.normal(size=(50, 5)); y = (rng.random(50) > 0.5).astype(int)
    s = sample_membership(z, y)
    assert 0 < s.min() and s.max() <= 1.0


def test_adaptive_weights():
    w = adaptive_weights({"a": 0.3, "b": 0.6, "c": 0.9}, tau=0.1)
    assert abs(sum(w.values()) - 1) < 1e-9 and w["a"] > w["b"] > w["c"]
    w_inf = adaptive_weights({"a": 0.3, "b": 0.9}, tau=1e9)
    assert abs(w_inf["a"] - 0.5) < 1e-6
