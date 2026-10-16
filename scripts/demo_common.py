"""Shared encoding for the walkthrough page (scripts/12_demo_export.py) and the live server (scripts/13_demo_server.py):
turns a TamesPredictor output into JSON with base64 images for every pipeline stage."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "demo" / "index.html"
DESC_LABEL = {
    "area_frac": ("area fraction", "nodule area / image area"),
    "taller_than_wide": ("taller-than-wide", "height / width of the nodule box (TI-RADS shape)"),
    "eccentricity": ("eccentricity", "elongation of the fitted ellipse"),
    "solidity": ("solidity", "area / convex-hull area; low = lobulated margin"),
    "irregularity": ("irregularity", "perimeter² / (4π·area) − 1; 0 for a circle"),
    "echo_ratio": ("echo ratio", "mean intensity inside / surrounding ring; low = hypoechoic"),
    "echo_mean": ("echo mean", "mean normalized intensity inside"),
    "echo_std": ("echo SD", "heterogeneity inside"),
    "echo_entropy": ("echo entropy", "texture entropy inside"),
    "bright_frac": ("bright-spot fraction", "echogenic foci proxy (microcalcification-like)"),
    "n_components": ("component count", "connected parts of the delineation"),
}
SKY_BGR, VERMILION_BGR = (233, 180, 86), (63, 123, 240)


def uri(img: np.ndarray, quality: int = 84, png: bool = False) -> str:
    ok, buf = cv2.imencode(".png" if png else ".jpg", img, [] if png else [cv2.IMWRITE_JPEG_QUALITY, quality])
    return ("data:image/png;base64," if png else "data:image/jpeg;base64,") + base64.b64encode(buf.tobytes()).decode()


def gray(a01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(a01 * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def magma(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [1, 99])
    return cv2.applyColorMap((np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


def fit(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]; s = side / max(h, w)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img


def encode(out: dict, descriptor_order: list[str], stats: dict) -> dict:
    img = out["image"]; H, W = img.shape
    raw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = 520 / max(H, W); th = max(2, round(2 / max(s, 1e-6) * 0.9))
    ann = raw.copy()
    cnts, _ = cv2.findContours(out["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ann, cnts, -1, VERMILION_BGR if out["mode"] == "expert" else SKY_BGR, th, cv2.LINE_AA)
    y0, y1, x0, x1 = out["crop_box"]
    cv2.rectangle(ann, (x0, y0), (x1 - 1, y1 - 1), (240, 240, 240), max(1, th // 2), cv2.LINE_AA)
    x = gray(out["x_crop"])
    otsu = x.copy(); fg = out["otsu_crop"]
    otsu[fg] = (0.62 * otsu[fg] + 0.38 * np.array(SKY_BGR)).astype(np.uint8)
    mc, _ = cv2.findContours(out["mask_crop"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(otsu, mc, -1, (255, 255, 255), 1, cv2.LINE_AA)
    images = {"raw": uri(fit(raw, 520)), "annotated": uri(fit(ann, 520)), "thumb": uri(fit(ann, 132), 78),
              "crop": uri(x), "mu": uri(magma(out["mu_crop"])), "F": uri(magma(out["F_crop"])), "otsu": uri(otsu)}
    if out["gradcam"] is not None:
        h = out["gradcam"]; heat = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        a = (0.85 * h)[..., None]; images["cam"] = uri((x * (1 - a) + heat * a).astype(np.uint8))
    desc = [{"key": k, "label": DESC_LABEL[k][0], "help": DESC_LABEL[k][1], "value": out["descriptors"][k], **stats[k]} for k in descriptor_order]
    emb = out["embedding"]
    return {"mode": out["mode"], "size": [int(H), int(W)], "crop_box": out["crop_box"], "q": round(out["q"], 4), "snr": round(out["snr"], 2),
            "images": images, "descriptors": desc, "embedding": [round(float(v), 4) for v in emb],
            "per_seed": out["per_seed"], "p_final": out["p_final"], "t_star": out["t_star"], "label": out["label"]}


def descriptor_order() -> list[str]:
    import pandas as pd
    s = pd.read_csv(ROOT / "runs/tn3k_fold0/explain_tames_seed0/shap_summary.csv", index_col=0)["ensemble"]
    order = [k for k in s.sort_values(ascending=False).index if k in DESC_LABEL]
    return order + [k for k in DESC_LABEL if k not in order]


def render(payload: dict, out_path: Path | None = None) -> str:
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__TAMES_DATA__", data)
    if out_path:
        out_path.write_text(html)
    return html
