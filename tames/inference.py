"""Single-image inference with the trained TAMES AI models, returning every intermediate stage.

Used by scripts/11_predict.py (text output), scripts/12_demo_export.py (static walkthrough page) and
scripts/13_demo_server.py (live upload UI). Nothing here trains or changes a model; the equations are
those of tames/fuzzy, tames/features and tames/models, called unchanged.

    pred = TamesPredictor("configs/default.yaml", automatic=False)
    out = pred.predict(gray_uint8_image, mask_bool_or_None)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

from .config import load_config
from .data import nodule_crop
from .features import DESCRIPTOR_NAMES, fuse, sonographic_descriptors
from .fuzzy import fuzzy_maps, otsu_binarize
from .models import get_device
from .models.cnn import load_model, predict
from .models.ensemble import ensemble_probability
from .pipeline import load_processed, tabular_block

LEARNERS = ("dl", "rf", "xgb", "fsvm")


class _GradCAM:
    """Grad-CAM on the last residual block, identical to scripts/07_explain.py (per-map normalization)."""

    def __init__(self, model):
        self.model, self.A, self.G = model, None, None
        layer = model.backbone.layer4
        layer.register_forward_hook(lambda m, i, o: setattr(self, "A", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "G", go[0]))

    def __call__(self, x, tab):
        self.model.zero_grad()
        logit = self.model(x, tab)
        logit.sum().backward()
        w = self.G.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.A).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        cam = cam - cam.flatten(1).min(dim=1)[0][:, None, None]
        cam = cam / (cam.flatten(1).max(dim=1)[0][:, None, None] + 1e-8)
        return cam.detach()[0].numpy()


class TamesPredictor:
    def __init__(self, config: str = "configs/default.yaml", automatic: bool = False, seeds=(0, 1, 2, 3, 4)):
        self.automatic = automatic
        self.run = "tn3k_automask" if automatic else "tn3k_fold0"
        overrides = [f"run_name={self.run}"]
        if automatic:
            overrides += ["paths.processed=data/processed_afmo", "image.mask_source=afmo"]
        self.cfg = cfg = load_config(config, overrides)
        self.rd = Path(cfg["paths"]["runs"]) / self.run
        proc = Path(cfg["paths"]["processed"])
        self.fz = json.load(open(proc / "fuzzy_params.json"))
        self.pre = tabular_block(load_processed(cfg))["pre"]
        R = json.load(open(self.rd / "ensemble_tames_seed0/results.json"))
        self.t_star = float(R["models"]["tames"]["threshold"]["t_star"])
        self.weights = {ps["seed"]: ps["weights"] for ps in R["per_seed"]}
        self.seeds = [s for s in seeds if s in self.weights]
        self.size, self.margin = int(cfg["image"]["size"]), float(cfg["image"]["crop_margin"])
        self.device = get_device(cfg.get("device", "auto"))
        meta = pd.read_csv(proc / "meta_train.csv")
        self.desc_stats = {}
        for k in DESCRIPTOR_NAMES:
            v = meta[k].astype(float)
            b, m = v[meta.label == 0].dropna(), v[meta.label == 1].dropna()
            self.desc_stats[k] = {"axis": [float(v.quantile(0.02)), float(v.quantile(0.98))],
                                  "benign": [float(b.quantile(q)) for q in (0.25, 0.5, 0.75)],
                                  "malignant": [float(m.quantile(q)) for q in (0.25, 0.5, 0.75)]}
        self._cnn, self._learners, self._cam = {}, {}, None

    def _models(self, s, d_tab):
        if s not in self._cnn:
            self._cnn[s] = load_model(self.cfg, 2, self.rd / f"cnn_tames_seed{s}/model.pt", self.device, d_tab=d_tab).eval()
            self._learners[s] = joblib.load(self.rd / f"learners_tames_seed{s}/learners.joblib")
        return self._cnn[s], self._learners[s]

    def predict(self, img: np.ndarray, mask: np.ndarray | None = None, cam: bool = True) -> dict:
        """img: 2-D uint8 grey image; mask: boolean nodule delineation or None (automatic Otsu(F))."""
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        maps = fuzzy_maps(img, self.fz, self.fz["snr0"])
        otsu = otsu_binarize(maps["F"], self.fz["polarity"])
        used_mask = mask.astype(bool) if mask is not None else otsu
        y0, y1, x0, x1 = nodule_crop(used_mask, self.margin)
        crop = lambda a: cv2.resize(a[y0:y1, x0:x1].astype(np.float32), (self.size, self.size), interpolation=cv2.INTER_AREA)
        x_c, mu_c, F_c = crop(maps["x"]), crop(maps["mu"]), crop(maps["F"])
        otsu_c = cv2.resize(otsu[y0:y1, x0:x1].astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask_c = cv2.resize(used_mask[y0:y1, x0:x1].astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_NEAREST).astype(bool)
        X = np.stack([x_c, F_c]).astype(np.float32)[None]
        desc = sonographic_descriptors(maps["x"], used_mask)
        S = np.array([[desc[k] for k in DESCRIPTOR_NAMES]], dtype=np.float64)
        s_t, obs = self.pre.transform(S)
        q_s = obs.mean(axis=1).astype(np.float32)
        q_i = np.array([maps["q"]], dtype=np.float32)
        T = np.concatenate([q_s[:, None] * s_t, q_i[:, None], q_s[:, None]], axis=1).astype(np.float32)
        per_seed, emb0 = [], None
        for s in self.seeds:
            cnn, L = self._models(s, T.shape[1])
            p_dl, h = predict(cnn, X, self.device, T)
            if emb0 is None:
                emb0 = h[0].astype(float)
            z = fuse(h, s_t, q_i, q_s)
            P = {"dl": p_dl, "rf": L["rf"].predict_proba(z)[:, 1], "xgb": L["xgb"].predict_proba(z)[:, 1], "fsvm": L["fsvm"].predict_proba(z)}
            pe = float(ensemble_probability(P, self.weights[s])[0])
            per_seed.append({"seed": s, **{k: float(P[k][0]) for k in LEARNERS}, "ensemble": pe,
                             "weights": {k: float(self.weights[s][k]) for k in LEARNERS}})
        p_final = float(np.mean([r["ensemble"] for r in per_seed]))
        heat = None
        if cam:
            if self._cam is None:
                m0 = load_model(self.cfg, 2, self.rd / f"cnn_tames_seed{self.seeds[0]}/model.pt", torch.device("cpu"), d_tab=T.shape[1]).eval()
                self._cam = _GradCAM(m0)
            heat = self._cam(torch.from_numpy(X), torch.from_numpy(T))
        return {"mode": "automatic" if self.automatic else "expert", "image": img, "mask": used_mask, "otsu": otsu,
                "crop_box": [int(y0), int(y1), int(x0), int(x1)], "q": float(maps["q"]), "snr": float(maps["snr"]),
                "x_crop": x_c, "mu_crop": mu_c, "F_crop": F_c, "otsu_crop": otsu_c, "mask_crop": mask_c,
                "descriptors": {k: float(desc[k]) for k in DESCRIPTOR_NAMES}, "embedding": emb0,
                "per_seed": per_seed, "p_final": p_final, "t_star": self.t_star,
                "label": "malignant" if p_final >= self.t_star else "benign", "gradcam": heat}
