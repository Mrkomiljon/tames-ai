#!/usr/bin/env python
"""Figure 1 — image flow of the TN3K release through the partitions (85 mm, single column).

Counts are read from data/processed/splits.csv when present, otherwise the released numbers
are used (3,493 → 2,879 development / 614 test; fold-0 train 2,303 / validation 576).
    python scripts/plots/fig1_flow.py [--out paper/figures]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle_diagram import (ACCENT, RULE, TEST_FILL, TINT, arrow, box, canvas, label, save,  # noqa: E402
                              setup_font)

ROOT = Path(__file__).resolve().parents[2]


def counts():
    c = {"all": 3493, "dev": 2879, "dev_m": 974, "test": 614, "test_m": 236,
         "train": 2303, "train_m": 779, "val": 576, "val_m": 195}
    f = ROOT / "data" / "processed" / "splits.csv"
    if f.exists():
        import pandas as pd
        d = pd.read_csv(f)
        g = d.groupby("split")["label"].agg(["count", "sum"])
        c.update(train=int(g.loc["train", "count"]), train_m=int(g.loc["train", "sum"]),
                 val=int(g.loc["val", "count"]), val_m=int(g.loc["val", "sum"]),
                 test=int(g.loc["test", "count"]), test_m=int(g.loc["test", "sum"]))
        c["dev"], c["dev_m"] = c["train"] + c["val"], c["train_m"] + c["val_m"]
        c["all"] = c["dev"] + c["test"]
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    a = ap.parse_args()
    setup_font()
    c = counts()
    pct = lambda m, n: f"{100 * m / n:.1f} %"
    T, L = 7.2, 6.0                      # title / line sizes (pt)
    W, H = 85, 92
    fig, ax = canvas(W, H)
    x0, full = 2, 81

    # level 1 — release
    y1, h1 = 2, 12.5
    box(ax, x0, y1, full, h1, f"TN3K public release · {c['all']:,} images",
        ["one B-mode image, one radiologist delineation and",
         "one benign/malignant label per image · no patient identifiers"], title_size=T, line_size=L)

    # level 2 — development | test
    y2, h2 = 22, 14
    dw, tw, g = 47.5, 31, 2.5
    dx, tx = x0, x0 + dw + g
    box(ax, dx, y2, dw, h2, "Development set",
        [f"{c['dev']:,} images · {c['dev_m']} malignant ({pct(c['dev_m'], c['dev'])})",
         "official fold-0 lists · every tuning decision"], title_size=T, line_size=L)
    box(ax, tx, y2, tw, h2, "Official test set",
        [f"{c['test']} images", f"{c['test_m']} malignant ({pct(c['test_m'], c['test'])})",
         "scored once, all choices frozen"], fill=TEST_FILL, edge=ACCENT, lw=0.9, title_size=T, line_size=L)

    # level 3 — training | validation (under development)
    y3, h3 = 44, 18
    sw = (dw - 2) / 2
    vx = dx + sw + 2
    box(ax, dx, y3, sw, h3, "Training",
        [f"{c['train']:,} images", f"{c['train_m']} malignant", "fits $f_\\theta$, the four learners,",
         "descriptor scaling"], title_size=T, line_size=L)
    box(ax, vx, y3, sw, h3, "Validation",
        [f"{c['val']} images", f"{c['val_m']} malignant", "selects $\\alpha,\\beta,\\gamma,\\lambda,\\tau,\\omega_k$,",
         "learner settings and $t^*$"], title_size=T, line_size=L)

    # level 4 — evaluation
    y4, h4 = 70, 16.5
    box(ax, x0, y4, full, h4, "Evaluation on the 614 test images",
        ["expert delineation (Tables 3–4) · automatic delineation (Table 5)",
         "image degradation (Table 6) · 10–50 % of training images (Table 7)",
         "calibration and explanations (Sections 5.8–5.9)"], fill=TINT, title_size=T, line_size=L)

    # connectors
    cx = x0 + full / 2
    ya = y1 + h1 + 3.5
    ax.plot([cx, cx], [y1 + h1, ya], color=RULE, lw=0.8)
    xa, xb = dx + dw / 2, tx + tw / 2
    ax.plot([xa, xb], [ya, ya], color=RULE, lw=0.8)
    arrow(ax, (xa, ya), (xa, y2)); arrow(ax, (xb, ya), (xb, y2))
    yb = y2 + h2 + 3.5
    ax.plot([xa, xa], [y2 + h2, yb], color=RULE, lw=0.8)
    xt, xv = dx + sw / 2, vx + sw / 2
    ax.plot([xt, xv], [yb, yb], color=RULE, lw=0.8)
    arrow(ax, (xt, yb), (xt, y3)); arrow(ax, (xv, yb), (xv, y3))
    # frozen models → evaluation; test images → evaluation
    yc = y3 + h3 + 3.5
    ax.plot([xt, xt], [y3 + h3, yc], color=RULE, lw=0.8)
    ax.plot([xv, xv], [y3 + h3, yc], color=RULE, lw=0.8)
    ax.plot([xt, xv], [yc, yc], color=RULE, lw=0.8)
    arrow(ax, (xa, yc), (xa, y4))
    label(ax, xa + 1.5, (yc + y4) / 2, "frozen models", ha="left", size=5.8)
    arrow(ax, (xb, y2 + h2), (xb, y4))
    label(ax, xb - 1.5, y4 - 3.0, "images and labels", ha="right", size=5.8)

    label(ax, x0, H - 2, "Image-level split: TN3K carries no patient identifiers.", ha="left",
          size=6.2, style="italic")
    print(save(fig, "fig1_flow", a.out))


if __name__ == "__main__":
    main()
