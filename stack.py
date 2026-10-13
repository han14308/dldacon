"""Stack all outputs/oof_*.csv with a non-negative linear blend (nested CV for an honest estimate)."""
import sys
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, ".")
from src.common import load_train, load_test, OUT, TARGET, N_FOLDS, rmse, write_submission  # noqa: E402


def fit_blend(X, y):
    m = LinearRegression(positive=True).fit(X, y)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=None, help="comma list; default = all oof_*.csv")
    ap.add_argument("--out", default="stack")
    args = ap.parse_args()
    tr, te = load_train(), load_test()
    y = tr[TARGET].values
    tags = args.tags.split(",") if args.tags else sorted(p.stem[4:] for p in OUT.glob("oof_*.csv"))
    O, T = [], []
    for t in tags:
        o = pd.read_csv(OUT / f"oof_{t}.csv").set_index("ID").loc[tr.ID, "pred"].values
        if np.isnan(o).any():
            print(f"skip {t}: incomplete oof")
            continue
        O.append(o)
        T.append(pd.read_csv(OUT / f"test_{t}.csv").set_index("ID").loc[te.ID, "pred"].values)
        print(f"{t:28s} CV RMSE {rmse(y, o):.3f}")
    tags = [t for t in tags if not np.isnan(pd.read_csv(OUT / f"oof_{t}.csv")["pred"]).any()]
    O, T = np.stack(O, 1), np.stack(T, 1)

    # nested CV estimate of the blend
    oof_blend = np.zeros(len(y))
    for k in range(N_FOLDS):
        trn, val = (tr.fold != k).values, (tr.fold == k).values
        oof_blend[val] = fit_blend(O[trn], y[trn]).predict(O[val])
    print(f"\nsimple mean      CV RMSE {rmse(y, O.mean(1)):.3f}")
    print(f"nested NNLS blend CV RMSE {rmse(y, oof_blend):.3f}")
    m = fit_blend(O, y)
    print("weights:", dict(zip(tags, np.round(m.coef_, 3))), "intercept", round(m.intercept_, 2))
    pred = m.predict(T)
    print("test pred stats: mean %.2f std %.2f min %.2f max %.2f" % (pred.mean(), pred.std(), pred.min(), pred.max()))
    print("submission ->", write_submission(args.out, te.ID, pred))


if __name__ == "__main__":
    main()
