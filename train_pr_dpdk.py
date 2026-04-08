"""
Softmax-style probabilistic regression with a U-Net backbone.
- Bins defined uniformly in target space (dP-like), then normalized with HadGEM train stats
- Per-bin sigma from local spacing (soft targets)
- TRAIN: HadGEM PPE (PR -> dPdK)
- VAL:   HadGEM internal val split
"""

import gc
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from unet import ProbUNet


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
use_amp = False #torch.backends.mps.is_available()
print(f"Using device: {device} | AMP: {use_amp}")


ensemble_size = 1
base_seed = 42
base_ch = 400
gn_groups = 1
k_size = 3
pdrop = 0.0
num_bins = 64
sigma_scale = 0.6
batch_train = 10
batch_val = 10
num_epochs = 5000
patience = 5
grad_clip = 1.0

dP_min = -700
dP_max = 1200

random.seed(base_seed)
np.random.seed(base_seed)
torch.manual_seed(base_seed)


base_model_name = (
    f"unet_ens_HG789_PR_dPdK_Softmax_unet6R_ch{base_ch}_k{k_size}_"
    f"128x_dPbins{num_bins}_gn{gn_groups}_dpmin{dP_min}_dPmax{dP_max}"
)
weights_dir = "/Users/ewellmeyer/Documents/research/weights"
ensemble_dir = os.path.join(weights_dir, base_model_name)
os.makedirs(ensemble_dir, exist_ok=True)

hadgem_dir = "/Users/ewellmeyer/Documents/research/HadGEM"
input_file = os.path.join(hadgem_dir, "GA789_PR_his_rg128.nc")
truth_file = os.path.join(hadgem_dir, "GA789_dPdK_rg128.nc")


class ClimateDataset(torch.utils.data.Dataset):
    def __init__(self, X, Ydist):
        X = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)
        Y = np.transpose(Ydist, (0, 3, 1, 2)).astype(np.float32)
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def make_soft_labels(y_norm, bin_centers_norm, sigma_bins_norm):
    diff = y_norm[..., 0:1] - bin_centers_norm.reshape(1, 1, 1, -1)
    Y = np.exp(-0.5 * (diff / (sigma_bins_norm.reshape(1, 1, 1, -1) + 1e-12)) ** 2)
    Y = (Y / (Y.sum(axis=-1, keepdims=True) + 1e-12)).astype(np.float32)
    return Y


def nll_from_probs(probs, y_soft):
    return -(y_soft * probs.clamp_min(1e-6).log()).sum(dim=1).mean()


print("Loading HadGEM data...")
ds_in = xr.open_dataset(input_file)
ds_tgt = xr.open_dataset(truth_file)

X_raw = ds_in["PR"].values[..., np.newaxis].astype(np.float32)
y_raw = ds_tgt["dPdK"].values[..., np.newaxis].astype(np.float32)

ds_in.close()
ds_tgt.close()


idx = np.arange(len(X_raw))
tr_idx, tmp_idx = train_test_split(idx, test_size=0.3, random_state=base_seed)
va_idx, te_idx = train_test_split(tmp_idx, test_size=0.2 / 0.3, random_state=base_seed)
np.savez(os.path.join(ensemble_dir, "data_splits.npz"), train=tr_idx, val=va_idx, test=te_idx)
print(f"HadGEM Splits: Train={len(tr_idx)}, Val={len(va_idx)}, Test={len(te_idx)}")

X_tr, y_tr = X_raw[tr_idx], y_raw[tr_idx]
X_va_hg, y_va_hg = X_raw[va_idx], y_raw[va_idx]
y_va_hg_tensor = torch.from_numpy(np.transpose(y_va_hg, (0, 3, 1, 2)).astype(np.float32))

del X_raw, y_raw
gc.collect()

Cx = X_tr.shape[-1]
x_mean = X_tr.reshape(-1, Cx).mean(axis=0)
x_std = X_tr.reshape(-1, Cx).std(axis=0).clip(1e-6)
y_mean = float(y_tr.mean())
y_std = float(y_tr.std().clip(1e-6))

print(f"HadGEM train stats: x={float(x_mean[0]):.4f}±{float(x_std[0]):.4f} | y={y_mean:.4f}±{y_std:.4f}")

X_tr_n = (X_tr - x_mean) / x_std
X_va_hg_n = (X_va_hg - x_mean) / x_std
y_tr_n = (y_tr - y_mean) / y_std
y_va_hg_n = (y_va_hg - y_mean) / y_std

with open(os.path.join(ensemble_dir, "norm_stats.json"), "w") as f:
    json.dump(
        {"x_mean": x_mean.tolist(), "x_std": x_std.tolist(), "y_mean": y_mean, "y_std": y_std},
        f,
        indent=2,
    )


dP_centers = np.linspace(dP_min, dP_max, num_bins, dtype=np.float32)
bin_centers = ((dP_centers - y_mean) / y_std).astype(np.float32)

diffs = np.diff(bin_centers)
spacing = np.r_[diffs[0], 0.5 * (diffs[:-1] + diffs[1:]), diffs[-1]]
sigma_bins = np.maximum(spacing * sigma_scale, 1e-4).astype(np.float32)

with open(os.path.join(ensemble_dir, "born_bins.json"), "w") as f:
    json.dump(
        {
            "num_bins": int(num_bins),
            "bin_centers_norm": bin_centers.tolist(),
            "bin_centers_dP": dP_centers.tolist(),
            "sigma_bins_norm": sigma_bins.tolist(),
            "definition": "uniform_dP",
            "dP_min": float(dP_min),
            "dP_max": float(dP_max),
            "sigma_scale": float(sigma_scale),
        },
        f,
        indent=2,
    )


print("Building soft labels...")
Ytr = make_soft_labels(y_tr_n, bin_centers, sigma_bins)
Yva_hg = make_soft_labels(y_va_hg_n, bin_centers, sigma_bins)
bin_centers_t = torch.as_tensor(bin_centers, dtype=torch.float32, device=device).view(1, -1, 1, 1)

train_ds = ClimateDataset(X_tr_n, Ytr)
val_hg_ds = ClimateDataset(X_va_hg_n, Yva_hg)
train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True)
val_hg_loader = DataLoader(val_hg_ds, batch_size=batch_val, shuffle=False)

del X_tr, X_va_hg, y_tr, y_va_hg
del X_tr_n, X_va_hg_n, y_tr_n, y_va_hg_n
del Ytr, Yva_hg
gc.collect()


for member in range(0, ensemble_size):
    print(f"\n==== Training member {member} ====")

    final_path = os.path.join(ensemble_dir, f"{base_model_name}_member{member}.pth")
    best_path = os.path.join(ensemble_dir, f"best_member{member}.pth")

    torch.manual_seed(base_seed + member)
    np.random.seed(base_seed + member)
    random.seed(base_seed + member)

    model = ProbUNet(1, base_ch, k_size, pdrop, num_bins, gn_groups=gn_groups).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    if os.path.exists(best_path):
        print(f"Resuming from {best_path}")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        best_val = ckpt.get("best_val_loss", best_val)

    epochs_bad = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        tr_loss, ntr = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="mps", dtype=torch.float16, enabled=use_amp):
                probs = model(xb)
                loss = nll_from_probs(probs, yb)

            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            bs = xb.size(0)
            tr_loss += float(loss.item()) * bs
            ntr += bs

        tr_loss /= max(ntr, 1)

        model.eval()
        va_hg_loss, va_hg_rmse, nhg = 0.0, 0.0, 0
        with torch.no_grad():
            idx0 = 0
            for xb, yb in val_hg_loader:
                xb, yb = xb.to(device), yb.to(device)
                bs = xb.size(0)

                with torch.autocast(device_type="mps", dtype=torch.float16, enabled=use_amp):
                    probs = model(xb)
                    loss = nll_from_probs(probs, yb)

                mu_n = (probs * bin_centers_t).sum(dim=1, keepdim=True)
                mu = mu_n * y_std + y_mean

                y_true = y_va_hg_tensor[idx0 : idx0 + bs].to(device)
                rmse = torch.sqrt((mu - y_true).pow(2).mean())

                va_hg_loss += float(loss.item()) * bs
                va_hg_rmse += float(rmse.item()) * bs
                nhg += bs
                idx0 += bs

        va_hg_loss /= max(nhg, 1)
        va_hg_rmse /= max(nhg, 1)
        sch.step(va_hg_loss)

        print(
            f"Epoch {epoch:04d} | "
            f"Train NLL={tr_loss:.5f} || "
            f"HG Val NLL={va_hg_loss:.5f} RMSE={va_hg_rmse:.5f}"
        )

        if va_hg_loss < best_val - 1e-6:
            best_val = va_hg_loss
            epochs_bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "best_val_loss": best_val,
                    "epoch": epoch,
                    "val_hg_loss": va_hg_loss,
                },
                best_path,
            )
            print("  -> checkpoint saved")
        else:
            epochs_bad += 1
            if epochs_bad >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    torch.save({"model": model.state_dict()}, final_path)
    print(f"Saved {final_path}")

print("\nTraining complete.")
