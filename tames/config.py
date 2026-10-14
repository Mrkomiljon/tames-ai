"""Configuration loading: YAML file + optional dotted overrides (key=value)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_paths(cfg)
    for item in overrides or []:
        key, _, value = item.partition("=")
        _set_dotted(cfg, key.strip(), yaml.safe_load(value))
    return cfg


def _resolve_paths(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for k, v in cfg.get("paths", {}).items():
        p = Path(v)
        cfg["paths"][k] = str(p if p.is_absolute() else ROOT / p)
    return cfg


def _set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def run_dir(cfg: dict) -> Path:
    p = Path(cfg["paths"]["runs"]) / cfg["run_name"]
    p.mkdir(parents=True, exist_ok=True)
    return p
