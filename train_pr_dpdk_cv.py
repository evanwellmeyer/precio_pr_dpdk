"""
Leave-one-PPE-out cross-validation training for Softmax ProbUNet.

Three folds: hold out GA7, GA8, or GA9 as test set.
For each fold:
  - Train on 80% of the two remaining PPEs (combined)
  - Validate on 20% of the two remaining PPEs (early stopping / LR scheduling)
  - Test PPE is fully withheld until post_pr_dpdk_cv.py
Trains `ensemble_size` members per fold with different random seeds.
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
use_amp = False
print(f"Using device: {device} | AMP: {use_amp}")


ensemble_size = 3
base_seed     = 42
base_ch       = 8
gn_groups     = 1
k_size        = 3
pdrop         = 0.0
num_bins      = 64
sigma_scale   = 0.6
batch_train   = 100
batch_val     = 80
num_epochs    = 5000
patience      = 20
grad_clip     = 1.0
val_fraction  = 0.2

dP_min = -700
dP_max = 1200

PPE_FAMILIES = ["GA7", "GA8", "GA9"]

hadgem_dir = "/Users/ewellmeyer/Documents/research/HadGEM"
input_file = os.path.join(hadgem_dir, "GA789_PR_his_rg128.nc")
truth_file = os.path.join(hadgem_dir, "GA789_dPdK_rg128.nc")
weights_dir = "/Users/ewellmeyer/Documents/research/weights"

base_model_name = (
    f"unet_cv_HG789_PR_dPdK_Softmax_unet6R_ch{base_ch}_k{k_size}_"
    f"128x_dPbins{num_bins}_gn{gn_groups}_dpmin{dP_min}_dPmax{dP_max}"
)


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
ds_in  = xr.open_dataset(input_file)
ds_tgt = xr.open_dataset(truth_file)

realizations = ds_in.realization.values
ppe_of = {r: r.split("_")[0] for r in realizations}   # e.g. 'GA7_42' -> 'GA7'

X_all = ds_in["PR"].values.astype(np.float32)[..., np.newaxis]    # (N, H, W, 1)
y_all = ds_tgt["dPdK"].values.astype(np.float32)[..., np.newaxis] # (N, H, W, 1)
ds_in.close(); ds_tgt.close()
print(f"Total members: {len(realizations)}")


for test_ppe in PPE_FAMILIES:
    print(f"\n{'='*60}")
    print(f"FOLD: test PPE = {test_ppe}")
    print(f"{'='*60}")

    fold_dir = os.path.join(weights_dir, base_model_name, f"fold_{test_ppe}")
    os.makedirs(fold_dir, exist_ok=True)

    train_ppas = [p for p in PPE_FAMILIES if p != test_ppe]

    test_mask  = np.array([ppe_of[r] == test_ppe for r in realizations])
    train_pool_mask = ~test_mask

    pool_idx = np.where(train_pool_mask)[0]
    test_idx = np.where(test_mask)[0]

    tr_idx, va_idx = train_test_split(pool_idx, test_size=val_fraction, random_state=base_seed)
    print(f"  Train: {len(tr_idx)}  Val: {len(va_idx)}  Test: {len(test_idx)}")

    # save split indices
    np.savez(
        os.path.join(fold_dir, "data_splits.npz"),
        train=tr_idx, val=va_idx, test=test_idx,
        train_ppas=np.array(train_ppas), test_ppe=np.array([test_ppe]),
    )

    X_tr = X_all[tr_idx]
    y_tr = y_all[tr_idx]
    X_va = X_all[va_idx]
    y_va = y_all[va_idx]

    # norm stats from training members only
    Cx = X_tr.shape[-1]
    x_mean = X_tr.reshape(-1, Cx).mean(axis=0)
    x_std  = X_tr.reshape(-1, Cx).std(axis=0).clip(1e-6)
    y_mean = float(y_tr.mean())
    y_std  = float(y_tr.std().clip(1e-6))
    print(f"  Norm: x={float(x_mean[0]):.4f}±{float(x_std[0]):.4f}  y={y_mean:.4f}±{y_std:.4f}")

    with open(os.path.join(fold_dir, "norm_stats.json"), "w") as f:
        json.dump({"x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
                   "y_mean": y_mean, "y_std": y_std}, f, indent=2)

    # bin setup
    dP_centers = np.linspace(dP_min, dP_max, num_bins, dtype=np.float32)
    bin_centers = ((dP_centers - y_mean) / y_std).astype(np.float32)
    diffs   = np.diff(bin_centers)
    spacing = np.r_[diffs[0], 0.5 * (diffs[:-1] + diffs[1:]), diffs[-1]]
    sigma_bins = np.maximum(spacing * sigma_scale, 1e-4).astype(np.float32)

    with open(os.path.join(fold_dir, "born_bins.json"), "w") as f:
        json.dump({
            "num_bins": int(num_bins),
            "bin_centers_norm": bin_centers.tolist(),
            "bin_centers_dP": dP_centers.tolist(),
            "sigma_bins_norm": sigma_bins.tolist(),
            "definition": "uniform_dP",
            "dP_min": float(dP_min), "dP_max": float(dP_max),
            "sigma_scale": float(sigma_scale),
        }, f, indent=2)

    # normalize
    X_tr_n = (X_tr - x_mean) / x_std
    X_va_n = (X_va - x_mean) / x_std
    y_tr_n = (y_tr - y_mean) / y_std
    y_va_n = (y_va - y_mean) / y_std

    bin_centers_t = torch.as_tensor(bin_centers, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    y_va_tensor   = torch.from_numpy(np.transpose(y_va, (0, 3, 1, 2)).astype(np.float32))

    Ytr = make_soft_labels(y_tr_n, bin_centers, sigma_bins)
    Yva = make_soft_labels(y_va_n, bin_centers, sigma_bins)

    train_loader = DataLoader(ClimateDataset(X_tr_n, Ytr), batch_size=batch_train, shuffle=True)
    val_loader   = DataLoader(ClimateDataset(X_va_n, Yva), batch_size=batch_val,   shuffle=False)

    del X_tr, X_va, y_tr, y_va, X_tr_n, X_va_n, y_tr_n, y_va_n, Ytr, Yva
    gc.collect()

    for member in range(ensemble_size):
        print(f"\n  -- Member {member} --")
        final_path = os.path.join(fold_dir, f"{base_model_name}_member{member}.pth")
        best_path  = os.path.join(fold_dir, f"best_member{member}.pth")

        torch.manual_seed(base_seed + member)
        np.random.seed(base_seed + member)
        random.seed(base_seed + member)

        model = ProbUNet(1, base_ch, k_size, pdrop, num_bins, gn_groups=gn_groups).to(device)
        opt   = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
        sch   = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

        best_val   = float("inf")
        epochs_bad = 0

        if os.path.exists(best_path):
            print(f"    Resuming from {best_path}")
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            best_val = ckpt.get("best_val_loss", best_val)

        for epoch in range(1, num_epochs + 1):
            model.train()
            tr_loss, ntr = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                probs = model(xb)
                loss  = nll_from_probs(probs, yb)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                tr_loss += float(loss.item()) * xb.size(0)
                ntr     += xb.size(0)
            tr_loss /= max(ntr, 1)

            model.eval()
            va_loss, va_rmse, nva = 0.0, 0.0, 0
            with torch.no_grad():
                idx0 = 0
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    bs    = xb.size(0)
                    probs = model(xb)
                    loss  = nll_from_probs(probs, yb)
                    mu_n  = (probs * bin_centers_t).sum(dim=1, keepdim=True)
                    mu    = mu_n * y_std + y_mean
                    y_true = y_va_tensor[idx0 : idx0 + bs].to(device)
                    rmse  = torch.sqrt((mu - y_true).pow(2).mean())
                    va_loss  += float(loss.item()) * bs
                    va_rmse  += float(rmse.item()) * bs
                    nva      += bs
                    idx0     += bs
            va_loss  /= max(nva, 1)
            va_rmse  /= max(nva, 1)
            sch.step(va_loss)

            print(
                f"    Epoch {epoch:04d} | Train NLL={tr_loss:.5f} || "
                f"Val NLL={va_loss:.5f} RMSE={va_rmse:.5f}"
            )

            if va_loss < best_val - 1e-6:
                best_val   = va_loss
                epochs_bad = 0
                torch.save({
                    "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "best_val_loss": best_val, "epoch": epoch,
                }, best_path)
                print("    -> checkpoint saved")
            else:
                epochs_bad += 1
                if epochs_bad >= patience:
                    print(f"    Early stop at epoch {epoch}")
                    break

        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device)["model"])
        torch.save({"model": model.state_dict()}, final_path)
        print(f"    Saved {final_path}")

    gc.collect()

print("\nCV training complete.")
