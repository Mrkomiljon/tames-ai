"""Out-of-distribution gate for single-image inference: decides whether an uploaded image looks like the
B-mode thyroid ultrasound frames TAMES AI was trained on, before any prediction is shown.

The diagnostic models cannot do this themselves: a classifier trained on two classes maps any input onto
one of them, and its own embedding collapses unfamiliar images onto familiar ones. The gate therefore uses
a separate, generic feature space (ImageNet ResNet-18, average-pooled, L2-normalized) on the whole frame
and measures the mean cosine distance to the k nearest TN3K training images. Thresholds come from the
validation partition only. Nothing in the diagnostic models changes.

    gate = OODGate.load()                 # runs/ood/reference.npz, built by scripts/14_fit_ood.py
    verdict = gate.check(image_bgr_or_gray)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "runs" / "ood" / "reference.npz"
K = 5
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor:
    def __init__(self):
        net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        net.fc = torch.nn.Identity()
        self.dev = _device()
        self.net = net.eval().to(self.dev)

    def __call__(self, grays: list[np.ndarray]) -> np.ndarray:
        X = np.stack([cv2.resize(g, (224, 224), interpolation=cv2.INTER_AREA) for g in grays]).astype(np.float32) / 255.0
        x = (torch.from_numpy(X)[:, None].repeat(1, 3, 1, 1) - _MEAN) / _STD
        with torch.no_grad():
            f = self.net(x.to(self.dev)).cpu().numpy()
        return f / np.linalg.norm(f, axis=1, keepdims=True)


def colour_deviation(img: np.ndarray) -> float:
    """Mean |RGB − grey| per pixel; exactly 0 for every TN3K frame (pure greyscale B-mode)."""
    if img.ndim == 2:
        return 0.0
    c = img.astype(np.float32)
    return float(np.abs(c - c.mean(axis=2, keepdims=True)).mean())


def knn_distance(q: np.ndarray, ref: np.ndarray, k: int = K) -> np.ndarray:
    return 1.0 - np.sort(q @ ref.T, axis=1)[:, -k:].mean(axis=1)


class OODGate:
    def __init__(self, ref: np.ndarray, warn: float, reject: float, val_dist: np.ndarray):
        self.ref, self.warn, self.reject, self.val_dist = ref.astype(np.float32), float(warn), float(reject), np.sort(val_dist)
        self.fx = FeatureExtractor()

    @classmethod
    def load(cls, path: Path = REF) -> "OODGate":
        z = np.load(path)
        return cls(z["train"], float(z["warn"]), float(z["reject"]), z["val_dist"])

    def check(self, img: np.ndarray) -> dict:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        d = float(knn_distance(self.fx([gray]), self.ref)[0])
        col = colour_deviation(img)
        pct = float(100.0 * np.searchsorted(self.val_dist, d) / len(self.val_dist))
        reasons = []
        if d > self.reject:
            reasons.append("far")
        elif d > self.warn:
            reasons.append("unusual")
        if col > 2.0:
            reasons.append("colour")
        if gray.std() < 5:
            reasons.append("flat")
        status = "reject" if ("far" in reasons or "flat" in reasons) else "warn" if reasons else "ok"
        return {"status": status, "distance": round(d, 4), "warn": round(self.warn, 4), "reject": round(self.reject, 4),
                "val_percentile": round(pct, 1), "colour": round(col, 2), "reasons": reasons}


def fit(splits_csv: Path, out: Path = REF) -> dict:
    """Build the reference set from the training images and set thresholds on the validation images."""
    import pandas as pd
    sp = pd.read_csv(splits_csv)
    fx = FeatureExtractor()
    F = {}
    for split in ("train", "val", "test"):
        paths = sp[sp.split == split].image_path.tolist()
        F[split] = np.concatenate([fx([cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths[i:i + 64]]) for i in range(0, len(paths), 64)])
    dv, dt = knn_distance(F["val"], F["train"]), knn_distance(F["test"], F["train"])
    warn = float(np.percentile(dv, 99.5))            # 0.5 % of validation frames would get a warning
    reject = float(dv.max() * 1.25)                   # clearly beyond every validation frame
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, train=F["train"].astype(np.float16), warn=warn, reject=reject, val_dist=dv.astype(np.float32))
    summary = {"k": K, "feature_space": "ImageNet ResNet-18 avgpool (512-d), L2-normalized, whole frame",
               "warn_threshold": round(warn, 4), "reject_threshold": round(reject, 4),
               "val": {"n": int(len(dv)), "median": round(float(np.median(dv)), 4), "max": round(float(dv.max()), 4)},
               "test": {"n": int(len(dt)), "median": round(float(np.median(dt)), 4), "max": round(float(dt.max()), 4),
                        "warned": int(((dt > warn) & (dt <= reject)).sum()), "rejected": int((dt > reject).sum())}}
    json.dump(summary, open(out.parent / "ood_summary.json", "w"), indent=2)
    return summary
