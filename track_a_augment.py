"""Track A data expansion: extract features from K degraded copies (blur/noise/contrast/gamma)
of every train image. Labels are unchanged (hardness is independent of imaging quality), so this
multiplies the GBDT training set and teaches it to ignore quality-driven feature shifts.
-> outputs/features_train_aug.csv  (columns: ID, copy, <features>)"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

sys.path.insert(0, ".")
from src.common import load_train, read_gray, OUT  # noqa: E402
from src.features import extract  # noqa: E402

K = 4


def degrade_u8(u8: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    x = u8.astype(np.float32) / 255.0
    if rng.rand() < 0.6:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.8))
    if rng.rand() < 0.5:
        x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.4)
    if rng.rand() < 0.7:
        x = (x - 0.5) * rng.uniform(0.6, 1.3) + 0.5 + rng.uniform(-0.12, 0.12)
    if rng.rand() < 0.7:
        x = x + rng.randn(*x.shape).astype(np.float32) * rng.uniform(0.0, 0.08)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _one(args):
    id_, c = args
    rng = np.random.RandomState(int(id_[-6:]) * 10 + c)
    u8 = read_gray(id_)
    return {"ID": id_, "copy": c, **extract(degrade_u8(u8, rng))}


if __name__ == "__main__":
    ids = load_train()["ID"].tolist()
    jobs = [(i, c) for i in ids for c in range(1, K + 1)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(tqdm(ex.map(_one, jobs, chunksize=8), total=len(jobs), desc="aug"))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "features_train_aug.csv", index=False)
    print(df.shape)
