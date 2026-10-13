"""Track C (fast baseline): frozen DINOv2 ViT-S/14 embeddings (8x dihedral TTA) + Ridge regression."""
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import timm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, ".")
from src.common import load_train, load_test, read_gray, OUT, TARGET, N_FOLDS, rmse, save_preds, write_submission  # noqa: E402
from train_cnn import dihedral, MEAN, STD, DEV  # noqa: E402


@torch.no_grad()
def embed(model, ids, bs=32, img_size=252):
    feats = []
    for i in range(0, len(ids), bs):
        x = np.stack([read_gray(j) for j in ids[i:i + bs]]).astype(np.float32) / 255.0
        x = torch.from_numpy(x).to(DEV)
        x = ((x - MEAN) / STD).unsqueeze(1).repeat(1, 3, 1, 1)
        x = torch.nn.functional.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        acc = 0
        for k in range(8):
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                t = model.forward_features(dihedral(x, k)).float()
            cls, patch = t[:, 0], t[:, 1:].mean(1)
            acc = acc + torch.cat([cls, patch], 1)
        feats.append((acc / 8).cpu().numpy())
        print(f"  {min(i + bs, len(ids))}/{len(ids)}", end="\r")
    print()
    return np.concatenate(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--tag", default="c_dinov2_s")
    ap.add_argument("--img_size", type=int, default=252)
    args = ap.parse_args()
    tr, te = load_train(), load_test()
    cache = OUT / f"emb_{args.tag}.npz"
    if cache.exists():
        z = np.load(cache)
        Xtr, Xte = z["tr"], z["te"]
    else:
        model = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=args.img_size).to(DEV).eval()
        Xtr, Xte = embed(model, tr.ID.tolist(), img_size=args.img_size), embed(model, te.ID.tolist(), img_size=args.img_size)
        np.savez(cache, tr=Xtr, te=Xte)
    y = tr[TARGET].values
    print("embedding dim", Xtr.shape[1])

    best = None
    for alpha in (1, 3, 10, 30, 100, 300, 1000):
        oof = np.zeros(len(tr))
        for k in range(N_FOLDS):
            trn, val = (tr.fold != k).values, (tr.fold == k).values
            m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xtr[trn], y[trn])
            oof[val] = m.predict(Xtr[val])
        r = rmse(y, oof)
        print(f"alpha {alpha:5}: CV RMSE {r:.3f}")
        if best is None or r < best[0]:
            best = (r, alpha, oof)
    r, alpha, oof = best
    print(f"\nTrack C DINOv2 best alpha {alpha}: CV RMSE {r:.3f}")
    test_pred = np.zeros(len(te))
    for k in range(N_FOLDS):
        trn = (tr.fold != k).values
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xtr[trn], y[trn])
        test_pred += m.predict(Xte) / N_FOLDS
    save_preds(args.tag, tr.ID, oof, te.ID, test_pred)
    print("submission ->", write_submission(args.tag, te.ID, test_pred))


if __name__ == "__main__":
    main()
