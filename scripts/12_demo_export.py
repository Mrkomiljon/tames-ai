#!/usr/bin/env python
"""Step 12 — build the walkthrough page: run the trained models on selected test images in both settings
(expert and automatic delineation) and write a self-contained HTML (demo/tames_walkthrough.html).

    python scripts/12_demo_export.py [--ids te_0322 te_0268 ...] [--out demo/tames_walkthrough.html]

Default cases are the six test images of Figure 3 (two TN, one FP, two TP, one FN at t*). No training happens.
"""
import argparse, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from tames.data import load_image, load_mask
from tames.inference import TamesPredictor
from demo_common import ROOT, descriptor_order, encode, render

DEFAULT = ["te_0388", "te_0322", "te_0097", "te_0385", "te_0452", "te_0268"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ids", nargs="*", default=DEFAULT); ap.add_argument("--out", default=str(ROOT / "demo" / "tames_walkthrough.html"))
    a = ap.parse_args()
    sp = pd.read_csv(ROOT / "data/processed/splits.csv").set_index("id")
    order = descriptor_order()
    preds = {"expert": TamesPredictor(automatic=False), "automatic": TamesPredictor(automatic=True)}
    cases = []
    for cid in a.ids:
        row = sp.loc[cid]; img = load_image(row.image_path); mask = load_mask(row.mask_path)
        modes = {}
        for name, P in preds.items():
            out = P.predict(img, mask if name == "expert" else None)
            modes[name] = encode(out, order, P.desc_stats)
        cases.append({"id": cid, "truth": "malignant" if int(row.label) else "benign", "modes": modes})
        print(f"{cid}: truth {cases[-1]['truth']:9s} expert p {modes['expert']['p_final']:.3f} ({modes['expert']['label']}) · automatic p {modes['automatic']['p_final']:.3f} ({modes['automatic']['label']})")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    html = render({"cases": cases, "commit": commit, "generated": time.strftime("%Y-%m-%d")}, Path(a.out))
    print(f"wrote {a.out} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
