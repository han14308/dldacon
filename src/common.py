"""Shared utilities: paths, folds, image loading, metrics."""
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "open"
OUT = ROOT / "outputs"
SUB = ROOT / "submissions"
OUT.mkdir(exist_ok=True)
SUB.mkdir(exist_ok=True)

N_FOLDS = 5
FOLD_SEED = 42
TARGET = "hardness"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_train() -> pd.DataFrame:
    """train.csv with a fixed, stratified-by-target fold column (cached to outputs/folds.csv)."""
    fold_path = OUT / "folds.csv"
    if fold_path.exists():
        return pd.read_csv(fold_path)
    df = pd.read_csv(DATA / "train.csv")
    # stratify on target quantile bins so each fold sees the full hardness range
    bins = pd.qcut(df[TARGET], q=10, labels=False)
    rng = np.random.RandomState(FOLD_SEED)
    df["fold"] = -1
    for b in range(10):
        idx = np.where(bins == b)[0]
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            df.loc[i, "fold"] = k % N_FOLDS
    df.to_csv(fold_path, index=False)
    return df


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA / "sample_submission.csv")


def img_path(id_: str) -> Path:
    sub = "train" if id_.startswith("TRAIN") else "test"
    return DATA / sub / f"{id_}.png"


def read_gray(id_: str) -> np.ndarray:
    """uint8 HxW grayscale."""
    return np.array(Image.open(img_path(id_)).convert("L"))


def rmse(y, p) -> float:
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def save_preds(tag: str, ids_tr, oof, ids_te, test_pred):
    """Persist out-of-fold and test predictions for stacking."""
    pd.DataFrame({"ID": ids_tr, "pred": oof}).to_csv(OUT / f"oof_{tag}.csv", index=False)
    pd.DataFrame({"ID": ids_te, "pred": test_pred}).to_csv(OUT / f"test_{tag}.csv", index=False)


def write_submission(tag: str, ids_te, pred):
    sub = pd.DataFrame({"ID": ids_te, TARGET: pred})
    path = SUB / f"{tag}.csv"
    sub.to_csv(path, index=False)
    return path
