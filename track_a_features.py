"""Extract Track A features for train and test -> outputs/features_{train,test}.csv"""
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, ".")
from src.common import load_train, load_test, read_gray, OUT  # noqa: E402
from src.features import extract  # noqa: E402


def _one(id_):
    return {"ID": id_, **extract(read_gray(id_))}


def run(ids, name):
    with ProcessPoolExecutor() as ex:
        rows = list(tqdm(ex.map(_one, ids, chunksize=8), total=len(ids), desc=name))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"features_{name}.csv", index=False)
    print(name, df.shape)


if __name__ == "__main__":
    run(load_train()["ID"].tolist(), "train")
    run(load_test()["ID"].tolist(), "test")
