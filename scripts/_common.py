"""Shared CLI plumbing for scripts 03–07."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tames.config import load_config  # noqa: E402


def parse(desc: str, variant: bool = True, seeds: bool = True):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. cnn.epochs=5")
    if variant:
        ap.add_argument("--variant", default="tames", choices=["tames", "image_only", "abl_afmo_off", "abl_atcfm_off"])
    if seeds:
        ap.add_argument("--seeds", nargs="*", type=int, default=None, help="default: eval.seeds from the config")
    ap.add_argument("--quality-off", action="store_true", help="ablation (c): every Q_m = 1")
    return ap


def finish(args):
    cfg = load_config(args.config, args.set)
    seeds = args.seeds if getattr(args, "seeds", None) else list(cfg["eval"]["seeds"])
    return cfg, seeds
