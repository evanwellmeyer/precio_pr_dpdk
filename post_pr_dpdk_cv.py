"""
Post-analysis for leave-one-PPE-out cross-validation.

For each fold (test PPE = GA7, GA8, GA9):
  - Loads ensemble trained on the other two PPEs
  - Evaluates on the held-out test PPE
  - Applies member filtering (NaN/Inf + IQR outlier)
  - Reports RMSE vs PPE baseline for global and land-only

Also reports the mean across all three folds.
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from unet import ProbUNet


base_ch    = 8
gn_groups  = 1
k_size     = 3
num_bins   = 64
lat_dim    = 128
batch_size = 100
outlier_iqr_factor = 3.0

dP_min = -700
dP_max = 1200

PPE_FAMILIES = ["GA7", "GA8", "GA9"]

base_model_name = (
    f"unet_cv_HG789_PR_dPdK_Softmax_unet6R_ch{base_ch}_k{k_size}_"
    f"128x_dPbins{num_bins}_gn{gn_groups}_dpmin{dP_min}_dPmax{dP_max}"
)
weights_base = Path("/Users/ewellmeyer/Documents/research/weights") / base_model_name

data_dir      = "/Users/ewellmeyer/Documents/research/HadGEM"
input_file    = os.path.join(data_dir, f"GA789_PR_his_rg128.nc")
truth_file    = os.path.join(data_dir, f"GA789_dPdK_rg128.nc")
landmask_file = os.path.join(data_dir, "hadgem_landmask_rg128.nc")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


print("Loading HadGEM data...")
ds_input   = xr.open_dataset(input_file)
ds_truth   = xr.open_dataset(truth_file)
ds_lm      = xr.open_dataset(landmask_file)

realizations = ds_input.realization.values
ppe_of = {r: r.split("_")[0] for r in realizations}

X_all = ds_input.to_array().values  # (1, N, H, W) if single var, or use PR directly
# Handle both single-var and multi-var datasets
if "PR" in ds_input:
    X_all = ds_input["PR"].values.astype(np.float32)[:, np.newaxis, :, :]   # (N, 1, H, W)
else:
    X_all = ds_input.to_array().values.astype(np.float32)
    X_all = np.transpose(X_all, (1, 0, 2, 3))

if "dPdK" in ds_truth:
    y_all = ds_truth["dPdK"].values.astype(np.float32)   # (N, H, W)
else:
    y_all = ds_truth.to_array().values.astype(np.float32)
    y_all = y_all[0] if y_all.shape[0] == 1 else np.transpose(y_all, (1, 0, 2, 3))[:, 0]

landmask = ds_lm["land_mask"].values.astype(bool)   # (H, W)

lats = ds_input.latitude.values
lat_weights = np.cos(np.deg2rad(lats)).astype(np.float32)
lat_weights /= lat_weights.mean()

N, _, H, W = X_all.shape
denom_land = float((landmask * lat_weights[:, None]).sum() + 1e-12)

ds_input.close(); ds_truth.close(); ds_lm.close()


fold_results = {}

for test_ppe in PPE_FAMILIES:
    print(f"\n{'='*60}")
    print(f"FOLD: test PPE = {test_ppe}")
    print(f"{'='*60}")

    fold_dir = weights_base / f"fold_{test_ppe}"

    splits = np.load(fold_dir / "data_splits.npz", allow_pickle=True)
    train_idx = splits["train"]
    test_idx  = splits["test"]

    with open(fold_dir / "norm_stats.json") as f:
        ns = json.load(f)
    x_mean = np.array(ns["x_mean"], dtype=np.float32)
    x_std  = np.array(ns["x_std"],  dtype=np.float32)
    y_mean = float(ns["y_mean"])
    y_std  = float(ns["y_std"])

    with open(fold_dir / "born_bins.json") as f:
        bi = json.load(f)
    bin_centers = np.array(bi["bin_centers_norm"], dtype=np.float32)
    bin_centers_t = torch.as_tensor(bin_centers, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    # test set
    X_test = X_all[test_idx]                         # (Nt, 1, H, W)
    y_test = y_all[test_idx]                         # (Nt, H, W)
    X_test_n = (X_test - x_mean[None, :, None, None]) / x_std[None, :, None, None]
    Nt = len(test_idx)

    # PPE baseline: mean of training targets
    y_train = y_all[train_idx]                       # (Ntr, H, W)
    ppe_mean = y_train.mean(axis=0, keepdims=True)   # (1, H, W)

    diff_ppe = ppe_mean - y_test
    se_w_ppe = diff_ppe ** 2 * lat_weights[None, :, None]
    ppe_rmse_global = float(np.sqrt(se_w_ppe.mean(axis=(1, 2))).mean())
    ppe_rmse_land   = float(np.sqrt(
        (se_w_ppe * landmask[None]).sum(axis=(1, 2)) / denom_land
    ).mean())

    # load ensemble
    model_files = sorted(glob.glob(str(fold_dir / f"{base_model_name}_member*.pth")))
    n_ens = len(model_files)
    print(f"  Found {n_ens} ensemble members")

    models = []
    for m_idx, path in enumerate(model_files):
        model = ProbUNet(1, base_ch, k_size, 0.0, num_bins, gn_groups=gn_groups).to(device)
        ckpt  = torch.load(path, map_location=device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        missing, _ = model.load_state_dict(state, strict=False)
        if missing:
            print(f"    Member {m_idx}: {len(missing)} missing keys")
        model.eval()
        models.append(model)

    all_mu = np.zeros((n_ens, Nt, H, W), dtype=np.float32)

    n_batches = (Nt + batch_size - 1) // batch_size
    print(f"  Running inference ({n_batches} batches × {n_ens} members)...")
    for i in range(0, Nt, batch_size):
        bsz = min(batch_size, Nt - i)
        xb  = torch.as_tensor(X_test_n[i : i + bsz], dtype=torch.float32, device=device)
        for m_idx, model in enumerate(models):
            with torch.inference_mode():
                probs  = model.forward_components(xb).float()
                mu_n   = (probs * bin_centers_t).sum(dim=1)
                mu     = mu_n * y_std + y_mean
            all_mu[m_idx, i : i + bsz] = mu.cpu().numpy()

    # per-member RMSE
    diff_m  = all_mu - y_test[None]
    se_w_m  = diff_m ** 2 * lat_weights[None, None, :, None]
    member_rmse_global = np.sqrt(se_w_m.mean(axis=(2, 3))).mean(axis=1)   # (n_ens,)
    member_rmse_land   = np.sqrt(
        (se_w_m * landmask[None, None]).sum(axis=(2, 3)) / denom_land
    ).mean(axis=1)

    # member filtering
    nan_bad     = ~np.isfinite(member_rmse_global)
    med         = np.nanmedian(member_rmse_global)
    iqr         = np.nanpercentile(member_rmse_global, 75) - np.nanpercentile(member_rmse_global, 25)
    outlier_bad = member_rmse_global > med + outlier_iqr_factor * iqr
    bad         = nan_bad | outlier_bad
    good_idx    = np.where(~bad)[0]

    print(f"  Members retained: {len(good_idx)}/{n_ens}")
    for excl in np.where(bad)[0]:
        tags = []
        if nan_bad[excl]:     tags.append("NaN/Inf")
        if outlier_bad[excl]: tags.append(f"outlier RMSE={member_rmse_global[excl]:.2f}")
        print(f"    Excluded member {excl}: {', '.join(tags)}")

    # ensemble mean from good members
    ens_mu   = all_mu[good_idx].mean(axis=0)
    diff_ens = ens_mu - y_test
    se_w_ens = diff_ens ** 2 * lat_weights[None, :, None]
    ens_rmse_global = float(np.sqrt(se_w_ens.mean(axis=(1, 2))).mean())
    ens_rmse_land   = float(np.sqrt(
        (se_w_ens * landmask[None]).sum(axis=(1, 2)) / denom_land
    ).mean())

    imp_global = (1 - ens_rmse_global / (ppe_rmse_global + 1e-12)) * 100
    imp_land   = (1 - ens_rmse_land   / (ppe_rmse_land   + 1e-12)) * 100

    print(f"\n  Test PPE = {test_ppe}  (N={Nt})")
    print(f"  Global  | PPE baseline: {ppe_rmse_global:.4f}  Ens RMSE: {ens_rmse_global:.4f}  Improvement: {imp_global:.2f}%")
    print(f"  Land    | PPE baseline: {ppe_rmse_land:.4f}  Ens RMSE: {ens_rmse_land:.4f}  Improvement: {imp_land:.2f}%")

    fold_results[test_ppe] = {
        "ppe_rmse_global": ppe_rmse_global,
        "ppe_rmse_land":   ppe_rmse_land,
        "ens_rmse_global": ens_rmse_global,
        "ens_rmse_land":   ens_rmse_land,
        "imp_global":      imp_global,
        "imp_land":        imp_land,
        "n_test":          int(Nt),
        "n_good_members":  int(len(good_idx)),
        "good_members":    good_idx.tolist(),
    }

    np.savez_compressed(
        fold_dir / "cv_results.npz",
        all_mu=all_mu,
        y_test=y_test,
        good_members=good_idx,
        test_indices=test_idx,
        train_indices=train_idx,
        lat_weights=lat_weights,
        landmask=landmask,
    )
    with open(fold_dir / "cv_results.json", "w") as f:
        json.dump(fold_results[test_ppe], f, indent=2)


# summary across folds
print(f"\n{'='*60}")
print("CROSS-VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"{'Fold':<8} {'PPE base (G)':<16} {'Ens RMSE (G)':<16} {'Imp (G)':<10} {'PPE base (L)':<16} {'Ens RMSE (L)':<16} {'Imp (L)'}")
for ppe, r in fold_results.items():
    print(f"{ppe:<8} {r['ppe_rmse_global']:<16.4f} {r['ens_rmse_global']:<16.4f} {r['imp_global']:<10.2f} "
          f"{r['ppe_rmse_land']:<16.4f} {r['ens_rmse_land']:<16.4f} {r['imp_land']:.2f}")

if len(fold_results) == 3:
    mean_imp_g = np.mean([r["imp_global"] for r in fold_results.values()])
    mean_imp_l = np.mean([r["imp_land"]   for r in fold_results.values()])
    print(f"\n  Mean improvement — Global: {mean_imp_g:.2f}%   Land: {mean_imp_l:.2f}%")

with open(weights_base / "cv_summary.json", "w") as f:
    json.dump(fold_results, f, indent=2)

print("\nAnalysis complete.")
