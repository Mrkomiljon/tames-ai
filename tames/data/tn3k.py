"""TN3K dataset access: splits, image/mask loading, nodule-centered crops.

Layout expected under `paths.raw` (see scripts/00_download_tn3k.sh):
    trainval-image/NNNN.jpg   trainval-mask/NNNN.jpg   tn3k-trainval-foldK.json
    test-image/NNNN.jpg       test-mask/NNNN.jpg
and under `paths.labels`: label4trainval.csv, label4test.csv  (name,label; 0 benign, 1 malignant).

Splits: `train` and `val` are the official fold-K lists (indices into the
int-sorted trainval file names, as in the TRFE-Net loader); `test` is the official
test set. TN3K has no patient identifiers, so the split is image-level.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _read_labels(csv_path: Path) -> dict[str, int]:
    labels = {}
    for line in csv_path.read_text().strip().splitlines():
        name, lab = line.strip().split(",")
        labels[name] = int(lab)
    return labels


def load_split(raw: str | Path, labels_dir: str | Path, fold: int = 0) -> pd.DataFrame:
    raw, labels_dir = Path(raw), Path(labels_dir)
    lab_tv = _read_labels(labels_dir / "label4trainval.csv")
    lab_te = _read_labels(labels_dir / "label4test.csv")
    names_tv = sorted((raw / "trainval-image").glob("*.jpg"), key=lambda p: int(p.stem))
    names_te = sorted((raw / "test-image").glob("*.jpg"), key=lambda p: int(p.stem))
    folds = json.load(open(raw / f"tn3k-trainval-fold{fold}.json"))
    rows = []
    for split, idx in (("train", folds["train"]), ("val", folds["val"])):
        for i in idx:
            p = names_tv[i]
            rows.append({"id": f"tv_{p.stem}", "split": split, "image_path": str(p),
                         "mask_path": str(raw / "trainval-mask" / p.name), "label": lab_tv[p.name]})
    for p in names_te:
        rows.append({"id": f"te_{p.stem}", "split": "test", "image_path": str(p),
                     "mask_path": str(raw / "test-mask" / p.name), "label": lab_te[p.name]})
    df = pd.DataFrame(rows)
    assert df["id"].is_unique
    return df


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_mask(path: str | Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > 127


def nodule_crop(mask: np.ndarray, margin: float = 1.5, min_side: int = 64) -> tuple[int, int, int, int]:
    """Square crop (y0, y1, x0, x1) centered on the nodule bounding box, scaled by
    `margin`; falls back to the whole image when the mask is empty."""
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, h, 0, w
    cy, cx = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
    side = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * margin
    side = int(max(side, min_side))
    y0, x0 = int(round(cy - side / 2)), int(round(cx - side / 2))
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(h, y0 + side), min(w, x0 + side)
    return y0, y1, x0, x1
