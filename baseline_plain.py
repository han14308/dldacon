"""Independent sanity baseline: deliberately shares NO code with src/ or train_cnn.py.
torchvision resnet18 (ImageNet), resize 224, MSE, h/v flips only, no TTA, sklearn KFold."""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
from sklearn.model_selection import KFold

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda"
df = pd.read_csv("open/train.csv")
IMGS = {i: np.array(Image.open(f"open/train/{i}.png").convert("L").resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0 for i in df.ID}
print("loaded", len(IMGS), "images; y mean %.2f sd %.2f" % (df.hardness.mean(), df.hardness.std()))


class DS(Dataset):
    def __init__(self, ids, y, train):
        self.ids, self.y, self.train = list(ids), np.asarray(y, dtype=np.float32), train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        x = IMGS[self.ids[i]].copy()
        if self.train:
            if np.random.rand() < 0.5: x = x[:, ::-1]
            if np.random.rand() < 0.5: x = x[::-1, :]
        x = (x - 0.449) / 0.226
        x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).repeat(3, 1, 1)
        return x, torch.tensor(self.y[i])


def run(arch="resnet18", lr=1e-3, epochs=25, bs=32):
    y = df.hardness.values
    mu, sd = y.mean(), y.std()
    oof = np.zeros(len(df))
    for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=0).split(df)):
        m = getattr(models, arch)(weights="DEFAULT")
        m.fc = nn.Linear(m.fc.in_features, 1)
        m = m.to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        dl = DataLoader(DS(df.ID[tr], (y[tr] - mu) / sd, True), batch_size=bs, shuffle=True)
        dv = DataLoader(DS(df.ID[va], (y[va] - mu) / sd, False), batch_size=64)
        best = (1e9, None)
        for ep in range(epochs):
            m.train()
            for x, t in dl:
                x, t = x.to(DEV), t.to(DEV)
                loss = nn.functional.mse_loss(m(x).squeeze(1), t)
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
            m.eval(); p = []
            with torch.no_grad():
                for x, _ in dv: p.append(m(x.to(DEV)).squeeze(1).cpu().numpy())
            p = np.concatenate(p) * sd + mu
            r = float(np.sqrt(np.mean((p - y[va]) ** 2)))
            if r < best[0]: best = (r, p)
        oof[va] = best[1]
        print(f"fold {k}: best val RMSE {best[0]:.3f}", flush=True)
    print(f"{arch} lr={lr}: CV RMSE {np.sqrt(np.mean((oof - y) ** 2)):.3f}", flush=True)


if __name__ == "__main__":
    arch = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
    lr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
    run(arch, lr)
