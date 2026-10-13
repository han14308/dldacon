"""Tracks B/C: fine-tune a pretrained CNN for hardness regression.

Backbones
  micronet_resnet50  : NASA MicroNet ResNet50 (MIT, HF jstuckner/microscopy-resnet50-micronet)
  any timm name      : e.g. convnext_tiny.fb_in22k_ft_in1k, efficientnet_b0.ra_in1k, resnet50.a1_in1k

Only scale-preserving augmentations are used (flips / 90-degree rotations, mild
brightness-contrast-noise jitter): grain size is a key signal and must not be rescaled.
"""
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

sys.path.insert(0, ".")
from src.common import load_train, load_test, read_gray, OUT, TARGET, N_FOLDS, rmse, save_preds, write_submission, seed_everything  # noqa: E402

MEAN, STD = 0.449, 0.226  # ImageNet grayscale equivalents
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- data
def dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    """8 symmetries of the square. x: (..., H, W)."""
    if k >= 4:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k % 4, dims=[-2, -1])


import cv2


def make_channels(u8: np.ndarray) -> np.ndarray:
    """3 channels: raw, NLM-denoised, local std (7x7). Gives the net an explicit de-noised view."""
    raw = u8.astype(np.float32) / 255.0
    den = cv2.fastNlMeansDenoising(u8, None, h=12, templateWindowSize=7, searchWindowSize=21).astype(np.float32) / 255.0
    mu = cv2.blur(den, (7, 7)); mu2 = cv2.blur(den * den, (7, 7))
    lstd = np.sqrt(np.clip(mu2 - mu * mu, 0, None)) * 4.0
    return np.stack([raw, den, lstd], 0)


def degrade(x: torch.Tensor) -> torch.Tensor:
    """Random blur / noise / contrast / gamma. Target is independent of imaging quality,
    so this teaches invariance and effectively multiplies the training set."""
    x = x.unsqueeze(0).unsqueeze(0)
    if np.random.rand() < 0.5:
        sigma = np.random.uniform(0.3, 1.8)
        k = int(2 * round(2.5 * sigma) + 1)
        g = torch.exp(-(torch.arange(k, dtype=torch.float32) - k // 2) ** 2 / (2 * sigma ** 2)); g = g / g.sum()
        x = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k))
        x = F.conv2d(x, g.view(1, 1, k, 1))
    x = x[0, 0]
    if np.random.rand() < 0.5:
        x = x ** np.random.uniform(0.7, 1.4)
    if np.random.rand() < 0.7:
        x = (x - 0.5) * np.random.uniform(0.6, 1.3) + 0.5 + np.random.uniform(-0.12, 0.12)
    if np.random.rand() < 0.6:
        x = x + torch.randn_like(x) * np.random.uniform(0.0, 0.08)
    return x.clamp(0, 1)


class ImgDS(Dataset):
    def __init__(self, ids, y=None, train=False, degrade_p=0.0, chan3=False, crop=0):
        self.ids, self.y, self.train = list(ids), y, train
        self.degrade_p, self.chan3, self.crop = degrade_p, chan3, crop
        self.cache = {i: read_gray(i) for i in self.ids}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        u8 = self.cache[self.ids[i]]
        x = torch.from_numpy(u8).float() / 255.0
        if self.train:
            if np.random.rand() < self.degrade_p:
                x = degrade(x)
            else:
                if np.random.rand() < 0.7:
                    x = (x - 0.5) * np.random.uniform(0.8, 1.2) + 0.5 + np.random.uniform(-0.1, 0.1)
                if np.random.rand() < 0.3:
                    x = x + torch.randn_like(x) * np.random.uniform(0.0, 0.04)
                x = x.clamp(0, 1)
        if self.chan3:
            x = torch.from_numpy(make_channels((x.numpy() * 255).astype(np.uint8)))
            x = (x - MEAN) / STD
        else:
            x = ((x - MEAN) / STD).unsqueeze(0).repeat(3, 1, 1)
        if self.train:
            x = dihedral(x, np.random.randint(8))
            if self.crop:
                i0, j0 = np.random.randint(0, 256 - self.crop + 1, size=2)
                x = x[:, i0:i0 + self.crop, j0:j0 + self.crop]
        if self.y is None:
            return x
        return x, torch.tensor(self.y[i], dtype=torch.float32)


# ----------------------------------------------------------------------------- model
def load_micronet_resnet50():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("jstuckner/microscopy-resnet50-micronet", "resnet50_micronet_weights.pth")
    sd = torch.load(p, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m = timm.create_model("resnet50", pretrained=False, num_classes=0)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("fc.")]
    unexpected = [k for k in unexpected if not k.startswith("fc.")]
    print(f"[micronet] loaded; missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing e.g.", missing[:5])
    if unexpected:
        print("  unexpected e.g.", unexpected[:5])
    assert len(missing) < 10, "MicroNet weights did not map onto timm resnet50"
    return m


class Reg(nn.Module):
    def __init__(self, backbone: str, drop: float = 0.2):
        super().__init__()
        if backbone == "micronet_resnet50":
            self.body = load_micronet_resnet50()
        else:
            self.body = timm.create_model(backbone, pretrained=True, num_classes=0)
        nf = self.body.num_features
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(nf, 1))

    upscale = 1.0

    def forward(self, x):
        if self.upscale != 1.0:
            x = F.interpolate(x, scale_factor=self.upscale, mode="bilinear", align_corners=False)
        return self.head(self.body(x)).squeeze(-1)


# ----------------------------------------------------------------------------- train
CROP = 0


def _views(x):
    """Full image, or a 3x3 grid of CROPxCROP windows when crop training is on."""
    if not CROP:
        return [x]
    offs = np.linspace(0, 256 - CROP, 3).astype(int)
    return [x[:, :, i:i + CROP, j:j + CROP] for i in offs for j in offs]


@torch.no_grad()
def predict(model, loader, tta=True):
    model.eval()
    out = []
    for x in loader:
        x = x.to(DEV, non_blocking=True)
        ks = range(8) if tta else [0]
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            p = torch.stack([model(dihedral(v, k)).float() for k in ks for v in _views(x)]).mean(0)
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def run_fold(args, k, tr, te_ds, y_mean, y_std):
    seed_everything(args.seed + k)
    trn, val = tr[tr.fold != k], tr[tr.fold == k]
    ytr = ((trn[TARGET].values - y_mean) / y_std).astype(np.float32)
    yva = ((val[TARGET].values - y_mean) / y_std).astype(np.float32)
    dl_tr = DataLoader(ImgDS(trn.ID, ytr, train=True, degrade_p=args.degrade, chan3=args.chan3, crop=args.crop), batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True)
    dl_va = DataLoader(ImgDS(val.ID, chan3=args.chan3), batch_size=args.bs * 2, shuffle=False, num_workers=0)
    dl_te = DataLoader(te_ds, batch_size=args.bs * 2, shuffle=False, num_workers=0)

    model = Reg(args.backbone, args.drop).to(DEV)
    model.upscale = args.upscale
    head_params = list(model.head.parameters())
    body_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    opt = torch.optim.AdamW([{"params": body_params, "lr": args.lr}, {"params": head_params, "lr": args.lr * 5}],
                            weight_decay=args.wd)
    steps = args.epochs * len(dl_tr)
    warm = len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")

    best, best_val, best_te, best_ep = 1e9, None, None, -1
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, yb in dl_tr:
            x, yb = x.to(DEV, non_blocking=True), yb.to(DEV)
            if args.mixup > 0 and np.random.rand() < 0.8:
                lam = float(np.random.beta(args.mixup, args.mixup))
                perm = torch.randperm(len(x), device=DEV)
                x = lam * x + (1 - lam) * x[perm]
                yb = lam * yb + (1 - lam) * yb[perm]
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                p = model(x)
            loss = F.smooth_l1_loss(p.float(), yb, beta=1.0) if args.loss == "huber" else F.mse_loss(p.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(x)
        pv = predict(model, dl_va, tta=False)
        r = rmse(yva, pv) * y_std
        if r < best:
            best, best_ep = r, ep
            best_val = predict(model, dl_va, tta=True)
            best_te = predict(model, dl_te, tta=True)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  fold {k} ep {ep:3d} loss {tot/len(dl_tr.dataset):.4f} val RMSE {r:.3f} (best {best:.3f} @ {best_ep})")
    print(f"fold {k}: best val RMSE {best:.3f} @ epoch {best_ep} | TTA val RMSE {rmse(yva, best_val)*y_std:.3f}")
    return best_val * y_std + y_mean, best_te * y_std + y_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="micronet_resnet50")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--degrade", type=float, default=0.0, help="prob. of blur/noise/contrast degradation aug")
    ap.add_argument("--chan3", action="store_true", help="input = [raw, denoised, local-std] instead of raw x3")
    ap.add_argument("--loss", default="huber", choices=["huber", "mse"])
    ap.add_argument("--upscale", type=float, default=1.0, help="bilinear upsample factor applied to the input batch")
    ap.add_argument("--crop", type=int, default=0, help="random NxN crop in training; 3x3 grid average at eval")
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup beta alpha (0 = off)")
    args = ap.parse_args()
    tag = args.tag or args.backbone.split(".")[0].replace("/", "_")
    global CROP
    CROP = args.crop
    print(vars(args), "device", DEV)

    tr, te = load_train(), load_test()
    y_mean, y_std = tr[TARGET].mean(), tr[TARGET].std()
    te_ds = ImgDS(te.ID, chan3=args.chan3)
    oof = np.full(len(tr), np.nan)
    test_pred = np.zeros(len(te))
    folds = [int(f) for f in args.folds.split(",")]
    for k in folds:
        pv, pt = run_fold(args, k, tr, te_ds, y_mean, y_std)
        oof[tr.fold == k] = pv
        test_pred += pt / len(folds)
    m = ~np.isnan(oof)
    print(f"\n{tag} CV RMSE: {rmse(tr[TARGET].values[m], oof[m]):.3f}")
    save_preds(tag, tr.ID, oof, te.ID, test_pred)
    print("submission ->", write_submission(tag, te.ID, test_pred))


if __name__ == "__main__":
    main()
