"""
HadGEM post-analysis for Softmax NN (simple + memory-lean)

- Loads Softmax ensemble (logits -> softmax probabilities)
- Loads splits, normalization, bin metadata (uniform centers in normalized dP)
- Computes PPE baseline (mean over TRAIN dP targets)
- Evaluates RMSEs and saves compact arrays
- Filters bad ensemble members (NaN/Inf or IQR outlier)

Speed notes:
- All models loaded up front; single data pass (batch outer, members inner)
- Member mean predictions stored as all_mu (n_ens, N, H, W) float32 instead of
  accumulated probs — ensemble mean of mu == mu of ensemble mean probs, so no
  information is lost and we avoid the float16 overflow that caused NaN.
- All RMSE loops are vectorized over samples.
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from unet import ProbUNet


base_channels = 256
gn_groups = 1
kernel_size = 3
num_bins = 64
lat_dim = 128
batch_size = 100          # increase from 10; tune down if MPS OOMs
outlier_iqr_factor = 3.0  # exclude members whose mean global RMSE > median + N*IQR

dP_min = -700    # -700 dpdk ; -10 dpdp
dP_max = 1200     # 1200 dpdk ; 75 dpdpp

ens_name = (
    f"unet_ens_HG789_PR_dPdK_Softmax_unet6R_ch{base_channels}_k{kernel_size}_"
    f"{lat_dim}x_dPbins{num_bins}_gn{gn_groups}_dpmin{dP_min}_dPmax{dP_max}_sigma0.6"
)
ens_dir = Path("/Users/ewellmeyer/Documents/research/weights") / ens_name

split_ind_path = ens_dir / "data_splits.npz"
norm_stats_path = ens_dir / "norm_stats.json"
bin_info_path = ens_dir / "born_bins.json"

data_dir = "/Users/ewellmeyer/Documents/research/HadGEM"
input_file = os.path.join(data_dir, f"GA789_PR_his_rg{lat_dim}.nc")
truth_file = os.path.join(data_dir, f"GA789_dPdK_rg{lat_dim}.nc")
landmask_file = os.path.join(data_dir, "hadgem_landmask_rg128.nc")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


model_files = sorted(glob.glob(str(ens_dir / f"{ens_dir.name}_member*.pth")))
n_ens = len(model_files)
if n_ens == 0:
    raise RuntimeError("No ensemble members found.")
print(f"Found {n_ens} ensemble members")

splits = np.load(split_ind_path)
train_ind = splits["train"]
val_ind = splits["val"]
test_ind = splits["test"]

with open(norm_stats_path, "r") as f:
    ns = json.load(f)
x_mean = np.array(ns["x_mean"], dtype=np.float32)
x_std = np.array(ns["x_std"], dtype=np.float32)
y_mean = float(ns["y_mean"])
y_std = float(ns["y_std"])

with open(bin_info_path, "r") as f:
    bin_info = json.load(f)
assert int(bin_info["num_bins"]) == num_bins, "num_bins mismatch"

bin_centers = np.array(bin_info["bin_centers_norm"], dtype=np.float32)
bin_centers_t = torch.as_tensor(bin_centers, dtype=torch.float32, device=device).view(1, -1, 1, 1)


ds_input = xr.open_dataset(input_file)
ds_truth = xr.open_dataset(truth_file)

X = ds_input.to_array().values.astype(np.float32)
y = ds_truth.to_array().values.astype(np.float32)
X = np.transpose(X, (1, 0, 2, 3))
y = np.transpose(y, (1, 0, 2, 3))

ds_landmask = xr.open_dataset(landmask_file)
landmask = ds_landmask["land_mask"].values.astype(bool)  # (H, W)

lats = ds_input.latitude.values
lat_weights = np.cos(np.deg2rad(lats)).astype(np.float32)
lat_weights = lat_weights / lat_weights.mean()

X_norm = (X - x_mean[None, :, None, None]) / x_std[None, :, None, None]
y_norm = (y - y_mean) / y_std

N, _, H, W = X.shape
y_flat = y[:, 0, :, :]  # (N, H, W)
denom_land = float((landmask * lat_weights[:, None]).sum() + 1e-12)


# PPE baseline — fully vectorized, no per-sample loop
y_train_norm = y_norm[train_ind]
ppe_mean_norm = np.mean(y_train_norm, axis=0, keepdims=True)  # (1, 1, H, W)
ppe_mean_dP = ppe_mean_norm * y_std + y_mean                  # (1, 1, H, W)

diff_ppe = ppe_mean_dP[:, 0] - y_flat                         # (N, H, W)
se_w_ppe = diff_ppe ** 2 * lat_weights[None, :, None]
ppe_rmse_per_sample = np.sqrt(se_w_ppe.mean(axis=(1, 2)))
ppe_rmse_per_sample_land = np.sqrt(
    (se_w_ppe * landmask[None]).sum(axis=(1, 2)) / denom_land
)


# Load all models up front
print("Loading models...")
models = []
for m_idx, path in enumerate(model_files):
    model = ProbUNet(1, base_channels, kernel_size, 0.0, num_bins, gn_groups=gn_groups).to(device)
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Member {m_idx}: {len(missing)} missing keys in checkpoint")
    model.eval()
    models.append(model)


# Single data pass: batch loop outer, member loop inner.
# Store float32 mean predictions — avoids float16 overflow NaN and is 16× smaller
# than storing full prob distributions.
all_mu = np.zeros((n_ens, N, H, W), dtype=np.float32)

n_batches = (N + batch_size - 1) // batch_size
print(f"Running inference ({n_batches} batches × {n_ens} members)...")
for i in range(0, N, batch_size):
    bsz = min(batch_size, N - i)
    xb = torch.as_tensor(X_norm[i : i + bsz], dtype=torch.float32, device=device)

    for m_idx, model in enumerate(models):
        with torch.inference_mode():
            probs = model.forward_components(xb).float()       # (bsz, num_bins, H, W)
            mu_norm = (probs * bin_centers_t).sum(dim=1)       # (bsz, H, W)
            mu = mu_norm * y_std + y_mean
        all_mu[m_idx, i : i + bsz] = mu.cpu().numpy()

    print(f"  Batch {i // batch_size + 1}/{n_batches}")


# Vectorized per-member RMSE over all samples at once
diff_m = all_mu - y_flat[None]                                 # (n_ens, N, H, W)
se_w_m = diff_m ** 2 * lat_weights[None, None, :, None]
member_rmse_per_sample = np.sqrt(se_w_m.mean(axis=(2, 3)))     # (n_ens, N)
member_rmse_per_sample_land = np.sqrt(
    (se_w_m * landmask[None, None]).sum(axis=(2, 3)) / denom_land
)


# Member filtering: exclude NaN/Inf and IQR outliers
member_global_rmse = member_rmse_per_sample.mean(axis=1)       # (n_ens,)
nan_bad = ~np.isfinite(member_global_rmse)
med = np.nanmedian(member_global_rmse)
iqr = np.nanpercentile(member_global_rmse, 75) - np.nanpercentile(member_global_rmse, 25)
outlier_bad = member_global_rmse > med + outlier_iqr_factor * iqr
bad = nan_bad | outlier_bad
good_idx = np.where(~bad)[0]

print(f"\nMember filtering: {len(good_idx)}/{n_ens} members retained")
if bad.any():
    for excl in np.where(bad)[0]:
        tags = []
        if nan_bad[excl]:
            tags.append("NaN/Inf")
        if outlier_bad[excl]:
            tags.append(f"outlier RMSE={member_global_rmse[excl]:.2f}")
        print(f"  Excluded member {excl}: {', '.join(tags)}")


# Ensemble predictions from good members only
ens_mu = all_mu[good_idx].mean(axis=0)                         # (N, H, W)
diff_ens = ens_mu - y_flat
se_w_ens = diff_ens ** 2 * lat_weights[None, :, None]
nn_ens_rmse_per_sample = np.sqrt(se_w_ens.mean(axis=(1, 2)))
nn_ens_rmse_per_sample_land = np.sqrt(
    (se_w_ens * landmask[None]).sum(axis=(1, 2)) / denom_land
)


file_ids = np.arange(N)
print("\n" + "=" * 60)
print(f"RESULTS ({len(good_idx)}/{n_ens} Members):")
print("=" * 60)
for split_name, idx in [("Train", train_ind), ("Val", val_ind), ("Test", test_ind)]:
    good_sub = member_rmse_per_sample[np.ix_(good_idx, idx)]
    good_sub_land = member_rmse_per_sample_land[np.ix_(good_idx, idx)]

    print(f"\n{split_name} Set (Global):")
    print(f"  PPE Mean RMSE:       {ppe_rmse_per_sample[idx].mean():.4f}")
    print(f"  Softmax Ens RMSE:    {nn_ens_rmse_per_sample[idx].mean():.4f}")
    print(f"  Members RMSE:        {good_sub.mean():.4f} ± {good_sub.std():.4f}")
    imp = (1 - nn_ens_rmse_per_sample[idx].mean() / (ppe_rmse_per_sample[idx].mean() + 1e-12)) * 100
    print(f"  Improvement:         {imp:.2f}%")

    print(f"\n{split_name} Set (Land Only):")
    print(f"  PPE Mean RMSE:       {ppe_rmse_per_sample_land[idx].mean():.4f}")
    print(f"  Softmax Ens RMSE:    {nn_ens_rmse_per_sample_land[idx].mean():.4f}")
    print(f"  Members RMSE:        {good_sub_land.mean():.4f} ± {good_sub_land.std():.4f}")
    impL = (
        1 - nn_ens_rmse_per_sample_land[idx].mean() / (ppe_rmse_per_sample_land[idx].mean() + 1e-12)
    ) * 100
    print(f"  Improvement:         {impL:.2f}%")


results = {
    "file_ids": file_ids.tolist(),
    "good_members": good_idx.tolist(),
    "excluded_members": np.where(bad)[0].tolist(),
    "rmse_ppe": ppe_rmse_per_sample.tolist(),
    "rmse_softmax_mean": nn_ens_rmse_per_sample.tolist(),
    "rmse_softmax_members": member_rmse_per_sample.tolist(),
    "rmse_ppe_land": ppe_rmse_per_sample_land.tolist(),
    "rmse_softmax_mean_land": nn_ens_rmse_per_sample_land.tolist(),
    "rmse_softmax_members_land": member_rmse_per_sample_land.tolist(),
    "n_ensemble_members": int(n_ens),
    "n_good_members": int(len(good_idx)),
    "train_indices": train_ind.tolist(),
    "val_indices": val_ind.tolist(),
    "test_indices": test_ind.tolist(),
    "gn_groups": int(gn_groups),
}
with open(ens_dir / "softmax_ensemble_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

np.savez_compressed(
    ens_dir / "softmax_ensemble_analysis_arrays.npz",
    file_ids=file_ids,
    good_members=good_idx,
    rmse_ppe=ppe_rmse_per_sample,
    rmse_softmax_mean=nn_ens_rmse_per_sample,
    rmse_softmax_members=member_rmse_per_sample,
    rmse_ppe_land=ppe_rmse_per_sample_land,
    rmse_softmax_mean_land=nn_ens_rmse_per_sample_land,
    rmse_softmax_members_land=member_rmse_per_sample_land,
    lat_weights=lat_weights.astype(np.float32),
    landmask=landmask,
    bin_centers=bin_centers.astype(np.float32),
    train_indices=train_ind,
    val_indices=val_ind,
    test_indices=test_ind,
)

print("\nAnalysis complete.")
