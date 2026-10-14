#!/usr/bin/env python
"""Step 7 — explainability (Section 3.11).

SHAP: TreeSHAP for RF and XGBoost, KernelSHAP (train background) for the Fuzzy SVM and
the DL head; ensemble attribution φ_j^ens = Σ_k ω_k φ_j^k (exact, since p_ens is linear
in the learners). Rank stability of the ten most important features over bootstrap
resamples of the explained rows. Grad-CAM on the last convolutional block of f_θ with
deletion/insertion AUCs against a random-saliency control.

Writes runs/<run>/explain_<variant>_seed<k>/{shap_values.npz, shap_summary.csv,
shap_stability.json, shap_summary.png, gradcam.png, faithfulness.json}.

    python scripts/07_explain.py --config configs/default.yaml --variant tames --seeds 0
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from _common import finish, parse
from tames.config import run_dir
from tames.models import get_device
from tames.models.cnn import load_model
from tames.pipeline import VARIANTS, load_processed, stage_name, tabular_block

warnings.filterwarnings("ignore")


# -------------------------------------------------------------------- SHAP ---
def shap_all(cfg, variant, seed, out, log):
    import shap
    ldir = run_dir(cfg) / stage_name("learners", variant, seed)
    cdir = run_dir(cfg) / stage_name("cnn", variant, seed)
    edir = run_dir(cfg) / stage_name("ensemble", variant, 0)
    L = joblib.load(ldir / "learners.joblib")
    names, d_h = L["names"], L["d_h"]
    z_tr, z_te = np.load(ldir / "z_train.npy"), np.load(ldir / "z_test.npy")
    ex = cfg["explain"]
    rng = np.random.default_rng(seed)
    bg = z_tr[rng.permutation(len(z_tr))[: int(ex["shap_background"])]]
    rows = rng.permutation(len(z_te))[: int(ex["shap_nsamples"])]
    zs = z_te[rows]
    weights = json.load(open(edir / "results.json"))["per_seed"][0]["weights"]
    phi = {}
    log("[shap] RF (TreeSHAP)")
    sv = shap.TreeExplainer(L["rf"]).shap_values(zs)
    phi["rf"] = sv[1] if isinstance(sv, list) else sv[..., 1]
    log("[shap] XGB (TreeSHAP, probability output)")
    phi["xgb"] = shap.TreeExplainer(L["xgb"], data=bg, model_output="probability",
                                    feature_perturbation="interventional").shap_values(zs)
    log("[shap] FSVM (KernelSHAP)")
    phi["fsvm"] = shap.KernelExplainer(L["fsvm"].predict_proba, bg).shap_values(zs, nsamples=200, silent=True)
    log("[shap] DL head (KernelSHAP)")
    device = get_device(cfg.get("device", "auto"))
    d_tab = z_tr.shape[1] - d_h
    model = load_model(cfg, len(VARIANTS[variant]), cdir / "model.pt", device, d_tab).eval()

    def head_prob(z):
        with torch.no_grad():
            return torch.sigmoid(model.head_from_z(torch.from_numpy(np.asarray(z, np.float32)).to(device))).cpu().numpy()
    phi["dl"] = shap.KernelExplainer(head_prob, bg).shap_values(zs, nsamples=200, silent=True)
    phi = {k: np.asarray(v, dtype=np.float64).reshape(len(zs), -1) for k, v in phi.items()}
    phi["ensemble"] = sum(float(weights[k]) * phi[k] for k in weights)
    np.savez(out / "shap_values.npz", rows=rows, names=np.array(names), **phi)
    # summary: mean |φ| per feature (image embedding dims grouped) for the ensemble
    def grouped(v):
        g = {"image embedding (Σ|φ| over h_img)": np.abs(v[:, :d_h]).sum(axis=1).mean()}
        for j, n in enumerate(names[d_h:], start=d_h):
            g[n] = np.abs(v[:, j]).mean()
        return g
    summ = pd.DataFrame({k: grouped(v) for k, v in phi.items()}).sort_values("ensemble", ascending=False)
    summ.to_csv(out / "shap_summary.csv")
    log(summ.round(4).to_string())
    # rank stability of the top-10 features over bootstrap resamples (ensemble attribution)
    feats = list(summ.index)
    top10 = feats[:10]
    hits = {f: 0 for f in top10}
    B = 2000
    v = phi["ensemble"]
    for _ in range(B):
        idx = rng.integers(0, len(v), len(v))
        g = grouped(v[idx])
        order = sorted(g, key=g.get, reverse=True)[:10]
        for f in top10:
            hits[f] += f in order
    stab = {f: hits[f] / B for f in top10}
    json.dump({"top10": top10, "in_top10_frequency": stab, "n_rows": int(len(rows)), "n_boot": B},
              open(out / "shap_stability.json", "w"), indent=2)
    log(f"[shap] top-10 rank stability (fraction of resamples in top 10): "
        + ", ".join(f"{f}={s:.2f}" for f, s in stab.items()))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = summ["ensemble"].head(15)[::-1]
        plt.figure(figsize=(7, 5))
        plt.barh(top.index, top.values)
        plt.xlabel("mean |SHAP| (probability units), ensemble attribution")
        plt.tight_layout(); plt.savefig(out / "shap_summary.png", dpi=150); plt.close()
    except Exception as e:  # pragma: no cover
        log(f"[shap] plot skipped: {e}")


# ----------------------------------------------------------------- Grad-CAM ---
class GradCAM:
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
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        cam = cam - cam.flatten(1).min(dim=1)[0][:, None, None]
        cam = cam / (cam.flatten(1).max(dim=1)[0][:, None, None] + 1e-8)
        return torch.sigmoid(logit).detach(), cam.detach()


def _curve(model, x, tab, order, blurred, steps=20):
    """Deletion (replace top-saliency pixels by the blurred image) and insertion curves."""
    n = x.shape[-1] * x.shape[-2]
    dele, inse = [], []
    with torch.no_grad():
        for k in range(steps + 1):
            m = torch.zeros(n, device=x.device)
            m[order[: int(round(k / steps * n))]] = 1.0
            m = m.view(1, 1, *x.shape[-2:])
            dele.append(float(torch.sigmoid(model(x * (1 - m) + blurred * m, tab))))
            inse.append(float(torch.sigmoid(model(blurred * (1 - m) + x * m, tab))))
    return float(np.trapezoid(dele, dx=1 / steps)), float(np.trapezoid(inse, dx=1 / steps))


def gradcam_all(cfg, variant, seed, out, log):
    import cv2
    data = load_processed(cfg)
    chans = VARIANTS[variant]
    tab = tabular_block(data) if variant != "image_only" else None
    cdir = run_dir(cfg) / stage_name("cnn", variant, seed)
    device = get_device(cfg.get("device", "auto"))
    d_tab = 0 if tab is None else tab["test"]["tab"].shape[1]
    model = load_model(cfg, len(chans), cdir / "model.pt", device, d_tab).eval()
    cam = GradCAM(model)
    X = np.asarray(data["X"]["test"][:, chans], dtype=np.float32)
    y = data["y"]["test"]
    n = int(cfg["explain"]["n_gradcam"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[: max(n, 32)]
    del_c, ins_c, del_r, ins_r, panels = [], [], [], [], []
    for j, i in enumerate(idx):
        x = torch.from_numpy(X[i: i + 1]).to(device)
        t = None if tab is None else torch.from_numpy(tab["test"]["tab"][i: i + 1].astype(np.float32)).to(device)
        p, c = cam(x, t)
        c = c[0].cpu().numpy()
        blurred = torch.from_numpy(np.stack([cv2.GaussianBlur(X[i, k], (0, 0), 8) for k in range(len(chans))])[None]).to(device)
        order = np.argsort(-c.ravel())
        d, s = _curve(model, x, t, torch.from_numpy(order).to(device), blurred)
        dr, sr = _curve(model, x, t, torch.from_numpy(rng.permutation(c.size)).to(device), blurred)
        del_c.append(d); ins_c.append(s); del_r.append(dr); ins_r.append(sr)
        if j < n:
            img = (X[i, 0] * 255).astype(np.uint8)
            heat = cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET)
            over = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.55, heat, 0.45, 0)
            cv2.putText(over, f"y={int(y[i])} p={float(p):.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            panels.append(np.concatenate([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), over], axis=1))
    faith = {"deletion_auc_gradcam": float(np.mean(del_c)), "insertion_auc_gradcam": float(np.mean(ins_c)),
             "deletion_auc_random": float(np.mean(del_r)), "insertion_auc_random": float(np.mean(ins_r)),
             "n_images": int(len(idx)), "note": "lower deletion and higher insertion AUC than the random control = faithful"}
    json.dump(faith, open(out / "faithfulness.json", "w"), indent=2)
    log(f"[gradcam] deletion AUC {faith['deletion_auc_gradcam']:.3f} (random {faith['deletion_auc_random']:.3f}); "
        f"insertion AUC {faith['insertion_auc_gradcam']:.3f} (random {faith['insertion_auc_random']:.3f})")
    if panels:
        cv2.imwrite(str(out / "gradcam.png"), np.concatenate(panels, axis=0))


def main():
    args = parse(__doc__).parse_args()
    cfg, seeds = finish(args)
    seed = seeds[0]
    out = run_dir(cfg) / stage_name("explain", args.variant, seed)
    out.mkdir(parents=True, exist_ok=True)
    shap_all(cfg, args.variant, seed, out, print)
    gradcam_all(cfg, args.variant, seed, out, print)


if __name__ == "__main__":
    main()
