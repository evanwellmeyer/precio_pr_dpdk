"""
HadGEM post-analysis for Softmax NN (simple + memory-lean)

- Loads Softmax ensemble (logits -> softmax probabilities)
- Loads splits, normalization, bin metadata (uniform centers in normalized dP)
- Computes PPE baseline (mean over TRAIN dP targets)
- Evaluates RMSEs and saves compact arrays
- Filters bad ensemble members using validation-set NaN/Inf or IQR outliers

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
# from vit import ProbViT as ProbUNet


base_channels = 64
gn_groups = 1
kernel_size = 3
num_bins = 64
lat_dim = 128
batch_size = 64          # increase from 10; tune down if MPS OOMs
outlier_iqr_factor = 3  # exclude members whose validation mean global RMSE > median + N*IQR

dP_min = -700    # -700 dpdk ; -10 dpdp
dP_max = 1200     # 1200 dpdk ; 75 dpdpp

ens_name = (
    f"unet_ens_HG789_PR_dPdK_Softmax_unet6R_ch{base_channels}_k{kernel_size}_"
    f"{lat_dim}x_dPbins{num_bins}_gn{gn_groups}_dpmin{dP_min}_dPmax{dP_max}_sigma0.6"
)
ens_dir = Path("/Users/ewellmeyer/Documents/research/weights") / ens_name

# ens_dir = Path("/Users/ewellmeyer/Documents/research/weights/direct_pr2dP") / "ens_HG789_PR_dPdK_Softmax_unet6SR_1x_ch128_k3_128x_dPbins64_gn8"

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
global_weights = np.broadcast_to(lat_weights[None, :, None], (N, H, W)).astype(np.float32)
land_weights = (global_weights * landmask[None]).astype(np.float32)


def range_diagnostics(pred, truth, idx, weights, low, high):
    pred_sub = pred[idx].astype(np.float64)
    truth_sub = truth[idx].astype(np.float64)
    w = weights[idx].astype(np.float64)

    in_range = (truth_sub >= low) & (truth_sub <= high)
    se = (pred_sub - truth_sub) ** 2

    total_weight = float(w.sum())
    total_se = float((se * w).sum())
    rmse_all = float(np.sqrt(total_se / (total_weight + 1e-12)))

    w_in = w * in_range
    in_weight = float(w_in.sum())
    if in_weight > 0.0:
        rmse_in_range = float(np.sqrt(float((se * w_in).sum()) / in_weight))
    else:
        rmse_in_range = float("nan")

    out_range = ~in_range
    out_se = float((se * w * out_range).sum())
    pct_out_se = float(100.0 * out_se / (total_se + 1e-12))

    return {
        "rmse_all": rmse_all,
        "rmse_in_range_only": rmse_in_range,
        "pct_total_se_from_out_of_range": pct_out_se,
    }


def pct_improvement(model_rmse, baseline_rmse):
    model_rmse = np.asarray(model_rmse, dtype=np.float64)
    baseline_rmse = np.asarray(baseline_rmse, dtype=np.float64)
    improvement = (1.0 - model_rmse / (baseline_rmse + 1e-12)) * 100.0
    improvement = np.where(
        np.isfinite(model_rmse) & np.isfinite(baseline_rmse),
        improvement,
        np.nan,
    )
    if improvement.ndim == 0:
        return float(improvement)
    return improvement


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
ppe_pred = np.broadcast_to(ppe_mean_dP[:, 0], y_flat.shape).astype(np.float32)


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


# Member filtering: exclude NaN/Inf and IQR outliers using validation samples only.
filter_idx = val_ind
member_filter_rmse = member_rmse_per_sample[:, filter_idx].mean(axis=1)  # (n_ens,)
nan_bad = ~np.isfinite(member_filter_rmse)
med = np.nanmedian(member_filter_rmse)
iqr = np.nanpercentile(member_filter_rmse, 75) - np.nanpercentile(member_filter_rmse, 25)
filter_threshold = med + outlier_iqr_factor * iqr
outlier_bad = member_filter_rmse > filter_threshold
bad = nan_bad | outlier_bad
good_idx = np.where(~bad)[0]

print(f"\nMember filtering on validation set: {len(good_idx)}/{n_ens} members retained")
print(f"  Validation RMSE threshold: {filter_threshold:.4f}")
if bad.any():
    for excl in np.where(bad)[0]:
        tags = []
        if nan_bad[excl]:
            tags.append("NaN/Inf")
        if outlier_bad[excl]:
            tags.append(f"validation outlier RMSE={member_filter_rmse[excl]:.2f}")
        print(f"  Excluded member {excl}: {', '.join(tags)}")
if len(good_idx) == 0:
    raise RuntimeError("Member filtering removed every ensemble member.")


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
range_results = {}
for split_name, idx in [("Train", train_ind), ("Val", val_ind), ("Test", test_ind)]:
    good_sub = member_rmse_per_sample[np.ix_(good_idx, idx)]
    good_sub_land = member_rmse_per_sample_land[np.ix_(good_idx, idx)]
    sample_imp_global = pct_improvement(nn_ens_rmse_per_sample[idx], ppe_rmse_per_sample[idx])
    sample_imp_land = pct_improvement(
        nn_ens_rmse_per_sample_land[idx], ppe_rmse_per_sample_land[idx]
    )
    max_imp_global = float(np.nanmax(sample_imp_global))
    max_imp_land = float(np.nanmax(sample_imp_land))

    ppe_diag_global = range_diagnostics(ppe_pred, y_flat, idx, global_weights, dP_min, dP_max)
    ens_diag_global = range_diagnostics(ens_mu, y_flat, idx, global_weights, dP_min, dP_max)
    ppe_diag_land = range_diagnostics(ppe_pred, y_flat, idx, land_weights, dP_min, dP_max)
    ens_diag_land = range_diagnostics(ens_mu, y_flat, idx, land_weights, dP_min, dP_max)
    in_range_imp_global = pct_improvement(
        ens_diag_global["rmse_in_range_only"], ppe_diag_global["rmse_in_range_only"]
    )
    in_range_imp_land = pct_improvement(
        ens_diag_land["rmse_in_range_only"], ppe_diag_land["rmse_in_range_only"]
    )

    print(f"\n{split_name} Set (Global):")
    print(f"  PPE Mean RMSE:       {ppe_rmse_per_sample[idx].mean():.4f}")
    print(f"  Softmax Ens RMSE:    {nn_ens_rmse_per_sample[idx].mean():.4f}")
    print(f"  Members RMSE:        {good_sub.mean():.4f} ± {good_sub.std():.4f}")
    imp = (1 - nn_ens_rmse_per_sample[idx].mean() / (ppe_rmse_per_sample[idx].mean() + 1e-12)) * 100
    print(f"  Improvement:         {imp:.2f}%")
    print(f"  Max Improvement:     {max_imp_global:.2f}%")
    print(f"  Range:               [{dP_min:.0f}, {dP_max:.0f}]")
    print(
        "  PPE Range Diag:      "
        f"RMSE_all={ppe_diag_global['rmse_all']:.4f}  "
        f"RMSE_in_range_only={ppe_diag_global['rmse_in_range_only']:.4f}  "
        f"SE_out_of_range={ppe_diag_global['pct_total_se_from_out_of_range']:.2f}%"
    )
    print(
        "  Ens Range Diag:      "
        f"RMSE_all={ens_diag_global['rmse_all']:.4f}  "
        f"RMSE_in_range_only={ens_diag_global['rmse_in_range_only']:.4f}  "
        f"SE_out_of_range={ens_diag_global['pct_total_se_from_out_of_range']:.2f}%"
    )
    print(f"  In-Range Improvement:{in_range_imp_global:>11.2f}%")

    print(f"\n{split_name} Set (Land Only):")
    print(f"  PPE Mean RMSE:       {ppe_rmse_per_sample_land[idx].mean():.4f}")
    print(f"  Softmax Ens RMSE:    {nn_ens_rmse_per_sample_land[idx].mean():.4f}")
    print(f"  Members RMSE:        {good_sub_land.mean():.4f} ± {good_sub_land.std():.4f}")
    impL = (
        1 - nn_ens_rmse_per_sample_land[idx].mean() / (ppe_rmse_per_sample_land[idx].mean() + 1e-12)
    ) * 100
    print(f"  Improvement:         {impL:.2f}%")
    print(f"  Max Improvement:     {max_imp_land:.2f}%")
    print(
        "  PPE Range Diag:      "
        f"RMSE_all={ppe_diag_land['rmse_all']:.4f}  "
        f"RMSE_in_range_only={ppe_diag_land['rmse_in_range_only']:.4f}  "
        f"SE_out_of_range={ppe_diag_land['pct_total_se_from_out_of_range']:.2f}%"
    )
    print(
        "  Ens Range Diag:      "
        f"RMSE_all={ens_diag_land['rmse_all']:.4f}  "
        f"RMSE_in_range_only={ens_diag_land['rmse_in_range_only']:.4f}  "
        f"SE_out_of_range={ens_diag_land['pct_total_se_from_out_of_range']:.2f}%"
    )
    print(f"  In-Range Improvement:{in_range_imp_land:>11.2f}%")

    range_results[split_name.lower()] = {
        "range_low": float(dP_min),
        "range_high": float(dP_max),
        "global": {
            "ppe": ppe_diag_global,
            "ensemble": ens_diag_global,
            "max_pct_improvement": max_imp_global,
            "pct_improvement_in_range_only": in_range_imp_global,
        },
        "land_only": {
            "ppe": ppe_diag_land,
            "ensemble": ens_diag_land,
            "max_pct_improvement": max_imp_land,
            "pct_improvement_in_range_only": in_range_imp_land,
        },
    }


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
    "member_filtering": {
        "split": "validation",
        "index_key": "val_indices",
        "rmse_basis": "mean_global_rmse",
        "outlier_iqr_factor": float(outlier_iqr_factor),
        "median_rmse": float(med),
        "iqr_rmse": float(iqr),
        "threshold_rmse": float(filter_threshold),
        "member_rmse": member_filter_rmse.tolist(),
    },
    "train_indices": train_ind.tolist(),
    "val_indices": val_ind.tolist(),
    "test_indices": test_ind.tolist(),
    "gn_groups": int(gn_groups),
    "range_diagnostics": range_results,
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
    member_filter_rmse=member_filter_rmse,
    member_filter_threshold=np.float32(filter_threshold),
    member_filter_median=np.float32(med),
    member_filter_iqr=np.float32(iqr),
    member_filter_indices=filter_idx,
    lat_weights=lat_weights.astype(np.float32),
    landmask=landmask,
    bin_centers=bin_centers.astype(np.float32),
    train_indices=train_ind,
    val_indices=val_ind,
    test_indices=test_ind,
)

print("\nAnalysis complete.")
