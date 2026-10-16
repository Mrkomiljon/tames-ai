#!/usr/bin/env python
"""Step 14 — fit the out-of-distribution gate used by 11_predict.py and the walkthrough server (tames/ood.py).

    python scripts/14_fit_ood.py [--eval-dir folder_of_non_ultrasound_images]

Writes runs/ood/reference.npz (training features, thresholds) and runs/ood/ood_summary.json. With --eval-dir,
also reports how many images in that folder the gate rejects. No diagnostic model is touched.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from tames.ood import OODGate, ROOT, fit


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--eval-dir", default=None); a = ap.parse_args()
    s = fit(ROOT / "data/processed/splits.csv")
    print(json.dumps(s, indent=2))
    if a.eval_dir:
        gate = OODGate.load(); res = {}
        for p in sorted(Path(a.eval_dir).iterdir()):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                v = gate.check(img); res[p.name] = v["status"]; print(f"{p.name[:40]:40s} {v['status']:6s} distance {v['distance']:.3f} {v['reasons']}")
        print({k: list(res.values()).count(k) for k in ("ok", "warn", "reject")})


if __name__ == "__main__":
    main()
