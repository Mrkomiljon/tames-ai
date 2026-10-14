"""Image branch — Eq. (9) and the DL learner of Section 3.7.

A ResNet-18 backbone f_θ whose first convolution takes `in_ch` channels
([x̃, F] = 2 for TAMES; x̃ alone = 1 for the image-only baseline) produces an
embedding h_img ∈ R^{d_h}; a two-layer head g_φ (d_h → 64 → 1) gives p_DL.
Training uses binary cross-entropy, AdamW and early stopping on validation AUC.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_
import torchvision
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NoduleCNN(nn.Module):
    """f_θ (backbone + embedding layer) and the DL learner g_φ.

    With `d_tab > 0` the head receives the fused vector z = [q·h_img ; tab] where
    `tab` = [Q_struct·s̃, q, Q_struct] (Eq. 11), i.e. the DL learner is multimodal.
    With `d_tab = 0` the head sees h_img only (image-only baseline).
    """

    def __init__(self, in_ch: int = 2, d_h: int = 128, d_tab: int = 0, dropout: float = 0.3,
                 pretrained: bool = True, arch: str = "resnet18"):
        super().__init__()
        self.d_tab = int(d_tab)
        weights = "IMAGENET1K_V1" if pretrained else None
        try:
            net = getattr(torchvision.models, arch)(weights=weights)
        except Exception as e:  # no network for weights → train from scratch
            print(f"[cnn] pretrained weights unavailable ({e.__class__.__name__}); training from scratch")
            net = getattr(torchvision.models, arch)(weights=None)
        w = net.conv1.weight.data  # (64, 3, 7, 7)
        net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            net.conv1.weight.copy_(w.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1))
        n_feat = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.embed = nn.Sequential(nn.Linear(n_feat, d_h), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(d_h + self.d_tab, 64), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """h_img = f_θ([x̃, F]) — Eq. (9)."""
        return self.embed(self.backbone(x))

    def fused(self, h: torch.Tensor, tab: torch.Tensor | None) -> torch.Tensor:
        """z = [q·h ; tab] when tabular input is present (q is the last-but-one tab column)."""
        if self.d_tab == 0 or tab is None:
            return h
        q = tab[:, -2:-1]
        return torch.cat([q * h, tab], dim=1)

    def head_from_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z).squeeze(1)

    def forward(self, x: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        return self.head_from_z(self.fused(self.embedding(x), tab))


def build_cnn(cfg: dict, in_ch: int, d_tab: int = 0) -> NoduleCNN:
    c = cfg["cnn"]
    return NoduleCNN(in_ch=in_ch, d_h=int(c["d_h"]), d_tab=d_tab, dropout=float(c["dropout"]),
                     pretrained=bool(c["pretrained"]), arch=c["arch"])


class ArrayDataset(Dataset):
    """Tensors prepared by scripts/02_fuzzy_preprocess.py: X (n, C, H, W) float32 in [0,1];
    optional tabular rows `tab` (n, d_tab)."""

    def __init__(self, X: np.ndarray, y: np.ndarray | None, tab: np.ndarray | None = None,
                 augment: bool = False, rng_seed: int = 0):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32))
        self.tab = None if tab is None else torch.from_numpy(np.ascontiguousarray(tab, dtype=np.float32))
        self.augment = augment
        self.g = torch.Generator().manual_seed(rng_seed)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = _augment(x, self.g)
        t = torch.zeros(0) if self.tab is None else self.tab[i]
        if self.y is None:
            return x, t
        return x, t, self.y[i]


def _augment(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Horizontal flip, ±10° rotation, brightness ±10 % (Section 3.13)."""
    if torch.rand(1, generator=g).item() < 0.5:
        x = torch.flip(x, dims=[2])
    ang = (torch.rand(1, generator=g).item() * 20.0 - 10.0) * np.pi / 180.0
    cos, sin = float(np.cos(ang)), float(np.sin(ang))
    theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=torch.float32).unsqueeze(0)
    grid = F_.affine_grid(theta, (1, *x.shape), align_corners=False)
    x = F_.grid_sample(x.unsqueeze(0), grid, align_corners=False, padding_mode="border").squeeze(0)
    b = 1.0 + (torch.rand(1, generator=g).item() * 0.2 - 0.1)
    return torch.clamp(x * b, 0.0, 1.0)


@torch.no_grad()
def predict(model: NoduleCNN, X: np.ndarray, device: torch.device, tab: np.ndarray | None = None,
            batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probabilities, embeddings h_img) for an array of images."""
    model.eval()
    probs, embs = [], []
    for xb, tb in DataLoader(ArrayDataset(X, None, tab), batch_size=batch_size):
        xb = xb.to(device)
        tb = tb.to(device) if model.d_tab > 0 else None
        h = model.embedding(xb)
        p = torch.sigmoid(model.head_from_z(model.fused(h, tb)))
        probs.append(p.cpu().numpy())
        embs.append(h.cpu().numpy())
    return np.concatenate(probs), np.concatenate(embs)


def train_cnn(cfg: dict, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
              in_ch: int, device: torch.device, tab_tr: np.ndarray | None = None,
              tab_va: np.ndarray | None = None, seed: int = 0, log=print) -> tuple[NoduleCNN, dict]:
    """Train with early stopping on validation AUC; returns the best model and a history."""
    c = cfg["cnn"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    d_tab = 0 if tab_tr is None else int(tab_tr.shape[1])
    model = build_cnn(cfg, in_ch, d_tab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c["lr"]), weight_decay=float(c["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(c["epochs"]))
    pos = float(y_tr.mean())
    pos_weight = torch.tensor([(1.0 - pos) / max(pos, 1e-6)], device=device) if c.get("balance", True) else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(ArrayDataset(X_tr, y_tr, tab_tr, augment=True, rng_seed=seed),
                        batch_size=int(c["batch_size"]), shuffle=True,
                        drop_last=len(X_tr) > int(c["batch_size"]))
    best_auc, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(int(c["epochs"])):
        model.train()
        tot = 0.0
        for xb, tb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            tb = tb.to(device) if d_tab > 0 else None
            opt.zero_grad()
            loss = loss_fn(model(xb, tb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(xb)
        sched.step()
        p_va, _ = predict(model, X_va, device, tab_va)
        auc = roc_auc_score(y_va, p_va) if len(np.unique(y_va)) > 1 else float("nan")
        hist.append({"epoch": ep, "train_loss": tot / max(len(X_tr), 1), "val_auc": auc})
        log(f"  epoch {ep:3d}  loss {tot / max(len(X_tr), 1):.4f}  val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(c["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_auc": best_auc, "history": hist}


def save_model(model: NoduleCNN, path: Path) -> None:
    torch.save(model.state_dict(), path)


def load_model(cfg: dict, in_ch: int, path: Path, device: torch.device, d_tab: int = 0) -> NoduleCNN:
    m = build_cnn(cfg, in_ch, d_tab)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device)
