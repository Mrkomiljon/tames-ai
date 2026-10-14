"""One call that produces every fuzzy map of an image: x̃, c, σ, T, K, q, μ, O, C, F, w."""
from __future__ import annotations

import numpy as np

from .afmo import afmo
from .atcfm import (atcfm, local_contrast, local_entropy, local_stats, normalize,
                    quality_coefficient, snr)


def fuzzy_maps(img: np.ndarray, fz: dict, snr0: float, *, atcfm_on: bool = True,
               afmo_on: bool = True) -> dict:
    """Compute the ATCFM membership and AFMO map of one gray-scale image.

    `fz` is the `fuzzy:` block of the config (k, alpha, beta, r0, gamma, r_min, r_max, lam).
    `atcfm_on=False` is ablation (b): α = β = 0 and q = 1 (plain Gaussian membership).
    `afmo_on=False` is ablation (a): F is replaced by μ.
    """
    k = int(fz["k"])
    x = normalize(img)
    c, sigma = local_stats(x, k)
    sigma_g = float(x.std())
    T = local_entropy(x, k)
    K = local_contrast(x, k)
    s = snr(x)
    q = quality_coefficient(s, snr0)
    if atcfm_on:
        mu = atcfm(x, c, sigma, T, K, q, fz["alpha"], fz["beta"])
    else:
        mu = atcfm(x, c, sigma, T, K, 1.0, 0.0, 0.0)
    out = {"x": x, "c": c, "sigma": sigma, "sigma_g": sigma_g, "T": T, "K": K,
           "snr": s, "q": q, "mu": mu}
    if afmo_on:
        out.update(afmo(mu, sigma, sigma_g, fz["r0"], fz["gamma"], int(fz["r_min"]),
                        int(fz["r_max"]), fz["lam"]))
    else:
        out["F"] = mu.copy()
    return out
