#!/usr/bin/env python
"""Step 11 — inference on one B-mode image with the trained TAMES AI models (no training).

    python scripts/11_predict.py --image path/to/image.jpg [--mask path/to/mask.png] [--seeds 0 1 2 3 4]
                                 [--config configs/default.yaml] [--run tn3k_fold0] [--cam out.png]

Without --mask the crop and the descriptors come from the Otsu-binarized AFMO map (fully automatic
setting, Table 5 models under runs/tn3k_automask); with --mask they use the delineation (Table 3 models).
Prints the probability of every learner and the ensemble per seed, the seed-averaged p_final and the
label at the validation-fixed threshold t*. Optionally writes a Grad-CAM overlay of the seed-0 backbone.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, cv2, joblib, torch
from tames.config import load_config
from tames.fuzzy import fuzzy_maps, otsu_binarize
from tames.data import load_image, load_mask, nodule_crop
from tames.features import DESCRIPTOR_NAMES, fuse, sonographic_descriptors
from tames.pipeline import load_processed, tabular_block
from tames.models import get_device
from tames.models.cnn import load_model, predict
from tames.models.ensemble import ensemble_probability


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True); ap.add_argument("--mask", default=None)
    ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--run", default=None, help="run directory name; default tn3k_fold0 with --mask, tn3k_automask without")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4]); ap.add_argument("--cam", default=None, help="write a Grad-CAM overlay PNG")
    a = ap.parse_args()
    auto = a.mask is None
    run = a.run or ("tn3k_automask" if auto else "tn3k_fold0")
    cfg = load_config(a.config, [f"run_name={run}"] + (["paths.processed=data/processed_afmo", "image.mask_source=afmo"] if auto else []))
    rd = Path(cfg["paths"]["runs"]) / run
    fz = json.load(open(Path(cfg["paths"]["processed"]) / "fuzzy_params.json"))
    pre = tabular_block(load_processed(cfg))["pre"]              # descriptor statistics fitted on the training partition
    R = json.load(open(rd / "ensemble_tames_seed0/results.json")); t_star = R["models"]["tames"]["threshold"]["t_star"]
    dev = get_device(cfg.get("device", "auto")); size, margin = int(cfg["image"]["size"]), float(cfg["image"]["crop_margin"])

    img = load_image(a.image); m = fuzzy_maps(img, fz, fz["snr0"])
    mask = (load_mask(a.mask) > 0) if a.mask else otsu_binarize(m["F"], fz["polarity"])
    y0, y1, x0, x1 = nodule_crop(mask, margin)
    X = np.stack([cv2.resize(ch[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA) for ch in (m["x"], m["F"])]).astype(np.float32)[None]
    desc = sonographic_descriptors(m["x"], mask); S = np.array([[desc[k] for k in DESCRIPTOR_NAMES]], dtype=np.float64)
    s_t, obs = pre.transform(S); q_s = obs.mean(axis=1).astype(np.float32); q_i = np.array([m["q"]], dtype=np.float32)
    T = np.concatenate([q_s[:, None] * s_t, q_i[:, None], q_s[:, None]], axis=1).astype(np.float32)
    print(f"image {a.image}  ({img.shape[0]}×{img.shape[1]} px) · mask: {'radiologist' if a.mask else 'automatic Otsu(F)'} · crop y {y0}:{y1}, x {x0}:{x1} · q = {m['q']:.3f}")
    print("descriptors: " + ", ".join(f"{k} {desc[k]:.3f}" for k in DESCRIPTOR_NAMES))

    p_final = []
    for s in a.seeds:
        cnn = load_model(cfg, 2, rd / f"cnn_tames_seed{s}/model.pt", dev, d_tab=T.shape[1])
        L = joblib.load(rd / f"learners_tames_seed{s}/learners.joblib"); w = next(ps["weights"] for ps in R["per_seed"] if ps["seed"] == s)
        p_dl, h = predict(cnn, X, dev, T); z = fuse(h, s_t, q_i, q_s)
        P = {"dl": p_dl, "rf": L["rf"].predict_proba(z)[:, 1], "xgb": L["xgb"].predict_proba(z)[:, 1], "fsvm": L["fsvm"].predict_proba(z)}
        pe = float(ensemble_probability(P, w)[0]); p_final.append(pe)
        print(f"seed {s}: DL {float(p_dl[0]):.3f}  RF {float(P['rf'][0]):.3f}  XGB {float(P['xgb'][0]):.3f}  FSVM {float(P['fsvm'][0]):.3f}  →  ensemble {pe:.3f}   (ω = " + ", ".join(f"{k} {v:.2f}" for k, v in w.items()) + ")")
    pf = float(np.mean(p_final))
    print(f"\np_final (mean over {len(a.seeds)} seeds) = {pf:.3f}   t* = {t_star:.3f}   →  {'MALIGNANT' if pf >= t_star else 'benign'}")

    if a.cam:
        sys.path.insert(0, str(Path(__file__).resolve().parent)); from importlib import import_module
        cam = import_module("07_explain").GradCAM(load_model(cfg, 2, rd / f"cnn_tames_seed{a.seeds[0]}/model.pt", get_device("cpu"), d_tab=T.shape[1]).eval())
        _, heat = cam(torch.from_numpy(X), torch.from_numpy(T)); heat = heat[0].numpy()
        g = (X[0, 0] * 255).astype(np.uint8); hm = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        over = cv2.addWeighted(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), 1.0, hm, 0.0, 0); alpha = (0.6 * heat)[..., None]
        over = (over * (1 - alpha) + hm * alpha).astype(np.uint8)
        out = np.concatenate([cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), over], axis=1)
        cv2.putText(out, f"p={pf:.2f} ({'malignant' if pf >= t_star else 'benign'})", (size + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(a.cam, out); print("Grad-CAM overlay written to", a.cam)


if __name__ == "__main__":
    main()
