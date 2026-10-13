"""Track A: LightGBM on hand-crafted features. 5-fold CV x multiple seeds -> oof/test preds."""
import sys
import argparse

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, ".")
from src.common import load_train, load_test, OUT, TARGET, N_FOLDS, rmse, save_preds, write_submission  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tag", default="a_lgbm")
    ap.add_argument("--aug", action="store_true", help="also train on degraded-copy rows (outputs/features_train_aug.csv)")
    ap.add_argument("--objective", default="regression")
    args = ap.parse_args()

    tr = load_train().merge(pd.read_csv(OUT / "features_train.csv"), on="ID")
    te = load_test().merge(pd.read_csv(OUT / "features_test.csv"), on="ID")
    feats = [c for c in tr.columns if c not in ("ID", TARGET, "fold")]
    X, y = tr[feats].values, tr[TARGET].values
    Xte = te[feats].values
    print(f"{len(feats)} features, {len(tr)} train, {len(te)} test")

    params = dict(objective=args.objective, learning_rate=0.02, num_leaves=7,
                  min_child_samples=20, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=2.0, verbose=-1, n_estimators=4000)
    if args.objective == "huber":
        params["alpha"] = 1.5 * y.std()
    Xaug, yaug, faug = None, None, None
    if args.aug:
        A = pd.read_csv(OUT / "features_train_aug.csv").merge(tr[["ID", TARGET, "fold"]], on="ID")
        Xaug, yaug, faug = A[feats].values, A[TARGET].values, A["fold"].values
        print(f"aug rows: {len(A)}")

    oof = np.zeros(len(tr))
    test_pred = np.zeros(len(te))
    imp = pd.Series(0.0, index=feats)
    for seed in range(args.seeds):
        for k in range(N_FOLDS):
            trn, val = tr["fold"] != k, tr["fold"] == k
            m = lgb.LGBMRegressor(**params, random_state=seed)
            Xt, yt = X[trn], y[trn]
            if Xaug is not None:
                ma = faug != k
                Xt, yt = np.vstack([Xt, Xaug[ma]]), np.concatenate([yt, yaug[ma]])
            m.fit(Xt, yt, eval_set=[(X[val], y[val])], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
            oof[val] += m.predict(X[val]) / args.seeds
            test_pred += m.predict(Xte) / (args.seeds * N_FOLDS)
            imp += pd.Series(m.booster_.feature_importance("gain"), index=feats)
        print(f"seed {seed}: running oof RMSE = {rmse(y, oof * args.seeds / (seed + 1)):.3f}")

    print(f"\nTrack A CV RMSE: {rmse(y, oof):.3f}  (predict-mean baseline {y.std():.3f})")
    for k in range(N_FOLDS):
        v = tr["fold"] == k
        print(f"  fold {k}: {rmse(y[v], oof[v]):.3f}")
    print("\nTop-25 features by gain:")
    print((imp / imp.sum()).sort_values(ascending=False).head(25).round(4).to_string())

    save_preds(args.tag, tr["ID"], oof, te["ID"], test_pred)
    p = write_submission(args.tag, te["ID"], test_pred)
    print("submission ->", p)


if __name__ == "__main__":
    main()
