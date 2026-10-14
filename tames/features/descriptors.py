"""Structured sonographic descriptors of a nodule (the tabular modality on TN3K).

TN3K carries no laboratory or clinical variables, so the structured branch uses
TI-RADS-like descriptors computed from the image and the nodule delineation:
composition/echogenicity (intensity inside vs. a surrounding ring), shape
(taller-than-wide, eccentricity), margin (solidity, irregularity) and echogenic
foci (bright-spot fraction, a calcification proxy). All are NaN when the mask is
empty; the fusion step imputes them and lowers Q_struct accordingly (Eq. 10).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import label as cc_label, regionprops, shannon_entropy

DESCRIPTOR_NAMES = [
    "area_frac",        # nodule area / image area
    "taller_than_wide", # bbox height / width (TI-RADS shape)
    "eccentricity",
    "solidity",         # area / convex-hull area (margin smoothness)
    "irregularity",     # perimeter² / (4π·area) − 1 (0 for a circle)
    "echo_ratio",       # mean intensity inside / mean intensity in a surrounding ring
    "echo_mean",        # mean normalized intensity inside
    "echo_std",         # heterogeneity inside
    "echo_entropy",     # Shannon entropy inside
    "bright_frac",      # fraction of pixels inside > mean + 2 std (echogenic foci proxy)
    "n_components",     # number of connected components in the mask
]


def sonographic_descriptors(x: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """x: normalized gray image in [0, 1]; mask: boolean nodule mask of the same shape."""
    out = {k: np.nan for k in DESCRIPTOR_NAMES}
    if mask.sum() < 10:
        return out
    lab = cc_label(mask)
    props = sorted(regionprops(lab), key=lambda r: r.area, reverse=True)
    out["n_components"] = float(len(props))
    r = props[0]
    h, w = mask.shape
    out["area_frac"] = float(r.area) / float(h * w)
    y0, x0, y1, x1 = r.bbox
    out["taller_than_wide"] = float(y1 - y0) / float(max(x1 - x0, 1))
    out["eccentricity"] = float(r.eccentricity)
    out["solidity"] = float(r.solidity)
    out["irregularity"] = float(r.perimeter ** 2 / (4.0 * np.pi * r.area) - 1.0)
    inside = x[mask]
    ring = binary_dilation(mask, iterations=max(3, int(0.15 * np.sqrt(r.area)))) & ~mask
    out["echo_mean"] = float(inside.mean())
    out["echo_std"] = float(inside.std())
    out["echo_ratio"] = float(inside.mean() / (x[ring].mean() + 1e-6)) if ring.any() else np.nan
    out["echo_entropy"] = float(shannon_entropy(np.round(inside * 255).astype(np.uint8)))
    thr = inside.mean() + 2.0 * inside.std()
    out["bright_frac"] = float((inside > thr).mean())
    return out
