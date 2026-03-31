"""
HadGEM post-analysis for Softmax NN (simple + memory-lean)

- Loads Softmax ensemble (logits -> softmax probabilities)
- Loads splits, normalization, bin metadata (uniform centers in normalized dP)
- Computes PPE baseline (mean over TRAIN dP targets)
- Evaluates RMSEs and saves compact arrays (in PR space, mm/yr)
- NEW: Ranks members by validation performance and computes top-5 ensemble
"""

import os, glob, json
from pathlib import Path
import numpy as np
import xarray as xr
import torch
import torch.nn as nn

# --------------------------
# Config / paths
# --------------------------
architecture   = "unet6SR_05x"
base_channels  = 128         # must match training
gn_groups      = 8          # NEW: must match training GroupNorm setting
kernel_size    = 3
num_bins       = 64         # must match training
lat_dim        = 128
batch_size     = 10

dP_min = -750
dP_max = 1250

ens_dir = Path(
    f"/Users/ewellmeyer/Documents/research/weights/"
    f"ens_HG789_PR-dPdK_Softmax_{architecture}_ch{base_channels}_k{kernel_size}_{lat_dim}x_dPbins{num_bins}_gn{gn_groups}_dpmin{dP_min}_dPmax{dP_max}"
)

split_ind_path  = ens_dir / "data_splits.npz"
norm_stats_path = ens_dir / "norm_stats.json"
bin_info_path   = ens_dir / "born_bins.json"  # same filename as training script

data_dir       = "/Users/ewellmeyer/Documents/research/HadGEM"
input_file     = os.path.join(data_dir, f"GA789_PR_his_rg{lat_dim}.nc")
truth_file     = os.path.join(data_dir, f"GA789_dPdK_rg{lat_dim}.nc")
landmask_file  = os.path.join(data_dir, "hadgem_landmask_rg128.nc")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
use_amp = torch.backends.mps.is_available()
print(f"Using device: {device} | AMP: {use_amp}")

from models.unet import Unet6R

# --------------------------
# Model definition (Softmax version)
# --------------------------
class SoftmaxHead(nn.Module):
    """
    Softmax head:
    - Single real-valued logit per bin
    - Probabilities via softmax over bins
    """
    def __init__(self, in_channels, num_bins):
        super().__init__()
        self.num_bins = num_bins
        self.conv = nn.Conv2d(in_channels, num_bins, kernel_size=1)

    def forward_components(self, feat):
        # feat: [N, C, H, W]
        logits = self.conv(feat)              # [N, B, H, W]
        probs  = torch.softmax(logits, dim=1) # softmax over bins
        return probs  # single output now

class ProbUNet(nn.Module):
    def __init__(self, input_channels, base_channels, kernel_size, p_drop, num_bins, gn_groups):
        super().__init__()
        self.backbone = Unet6R(
            input_channels=input_channels,
            output_channels=base_channels,
            base_channels=base_channels,
            kernel_size=kernel_size,
            p_drop=p_drop,
            gn_groups=gn_groups,   
        )
        self.head = SoftmaxHead(base_channels, num_bins)

    def forward_components(self, x):
        feat = self.backbone(x)
        return self.head.forward_components(feat)

# --------------------------
# Load metadata
# --------------------------
model_files = sorted(glob.glob(str(ens_dir / f"{ens_dir.name}_member*.pth")))
n_ens = len(model_files)
print(f"Found {n_ens} ensemble members")

splits = np.load(split_ind_path)
train_ind = splits["train"]; val_ind = splits["val"]; test_ind = splits["test"]

with open(norm_stats_path, "r") as f:
    ns = json.load(f)
x_mean = np.array(ns["x_mean"], dtype=np.float32)  # PR mean
x_std  = np.array(ns["x_std"],  dtype=np.float32)
y_mean = float(ns["y_mean"])                       # dP mean
y_std  = float(ns["y_std"])

with open(bin_info_path, "r") as f:
    bi = json.load(f)
assert int(bi["num_bins"]) == num_bins, "num_bins mismatch"

# training saved centers under 'bin_centers_norm' (normalized dP)
bin_centers = np.array(bi["bin_centers_norm"], dtype=np.float32)
bin_centers_t = torch.as_tensor(bin_centers, dtype=torch.float32, device=device).view(1,-1,1,1)

# --------------------------
# Load data (PR and dP)
# --------------------------
ds_input = xr.open_dataset(input_file)   # PR (mm/yr)
ds_truth = xr.open_dataset(truth_file)   # dPdK (mm/yr per K, or similar)

X = ds_input.to_array().values.astype(np.float32)  # (var,N,H,W)
y = ds_truth.to_array().values.astype(np.float32)
X = np.transpose(X, (1,0,2,3))                     # (N,1,H,W)  historical PR
y = np.transpose(y, (1,0,2,3))                     # (N,1,H,W)  dP

ds_landmask = xr.open_dataset(landmask_file)
landmask = ds_landmask["land_mask"].values.astype(bool)  # (H,W)

lats = ds_input.latitude.values
lat_weights = np.cos(np.deg2rad(lats)).astype(np.float32)
lat_weights = lat_weights / lat_weights.mean()

# Normalize inputs and targets exactly as in training
X_norm = (X - x_mean[None,:,None,None]) / x_std[None,:,None,None]
y_norm = (y - y_mean) / y_std

# --------------------------
# PPE baseline (mean over TRAIN in normalized dP space)
# --------------------------
y_train_norm   = y_norm[train_ind]
ppe_mean_norm  = np.mean(y_train_norm, axis=0, keepdims=True)  # (1,1,H,W)
ppe_mean_dP    = ppe_mean_norm * y_std + y_mean                # dP baseline
PR_his_mean    = np.mean(X[train_ind], axis=0, keepdims=True)  # (1,1,H,W)
ppe_pred_PR    = PR_his_mean + ppe_mean_dP                     # PR future baseline

N, _, H, W = X.shape
accum_probs_sum = np.zeros((N, num_bins, H, W), dtype=np.float16)

member_rmse_per_sample      = np.zeros((n_ens, N), dtype=np.float32)
member_rmse_per_sample_land = np.zeros((n_ens, N), dtype=np.float32)

# --------------------------
# Evaluate ensemble
# --------------------------
for m_idx, path in enumerate(model_files):
    print(f"Processing member {m_idx} ...")
    model = ProbUNet(1, base_channels, kernel_size, 0.0, num_bins, gn_groups=gn_groups).to(device)
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.inference_mode(), torch.autocast(device_type="mps", dtype=torch.float16, enabled=use_amp):
        i = 0
        while i < N:
            bsz = min(batch_size, N - i)
            xb_norm = torch.tensor(X_norm[i:i+bsz], dtype=torch.float32, device=device)

            # Softmax model returns only probs over dP bins
            probs = model.forward_components(xb_norm)  # [B,N,H,W]

            probs_np = probs.detach().cpu().numpy().astype(np.float16)
            accum_probs_sum[i:i+bsz] += probs_np

            # Expected dP (normalized -> physical)
            mu_norm = (probs * bin_centers_t).sum(dim=1, keepdim=True)  # [bsz,1,H,W]
            mu      = mu_norm * y_std + y_mean                          # dP (mm/yr)

            mu_np = mu.detach().cpu().numpy()
            y_np  = y[i:i+bsz]   # true dP (mm/yr)

            wlat = lat_weights[None,None,:,None]

            for j in range(bsz):
                diff = mu_np[j:j+1] - y_np[j:j+1]
                se   = diff * diff
                se_w = se * wlat

                # global dP RMSE
                member_rmse_per_sample[m_idx, i+j] = float(np.sqrt(se_w.mean()))

                # land-only dP RMSE
                mask = landmask[None,None,:,:]
                se_w_mask = se_w * mask
                denom = float((landmask * lat_weights[:,None]).sum() + 1e-12)
                member_rmse_per_sample_land[m_idx, i+j] = float(
                    np.sqrt(se_w_mask.sum() / denom)
                )
            i += bsz

# --------------------------
# Ensemble mean probs -> dP RMSEs
# --------------------------
if n_ens == 0:
    raise RuntimeError("No ensemble members found.")
avg_probs = (accum_probs_sum / np.float16(n_ens)).astype(np.float32)

nn_ens_rmse_per_sample      = np.zeros(N, dtype=np.float32)
nn_ens_rmse_per_sample_land = np.zeros(N, dtype=np.float32)

i = 0
while i < N:
    bsz = min(batch_size, N - i)
    ap = avg_probs[i:i+bsz]  # (bsz, B, H, W)

    # Ensemble-mean expected dP
    mu_norm_np = (ap * bin_centers[None,:,None,None]).sum(axis=1, keepdims=True)
    mu_np      = mu_norm_np * y_std + y_mean   # dP (mm/yr)

    y_np = y[i:i+bsz]  # true dP (mm/yr)

    wlat = lat_weights[None,None,:,None]

    for j in range(bsz):
        diff = mu_np[j:j+1] - y_np[j:j+1]
        se   = diff * diff
        se_w = se * wlat

        # global dP RMSE
        nn_ens_rmse_per_sample[i+j] = float(np.sqrt(se_w.mean()))

        # land-only dP RMSE
        mask = landmask[None,None,:,:]
        se_w_mask = se_w * mask
        denom = float((landmask * lat_weights[:,None]).sum() + 1e-12)
        nn_ens_rmse_per_sample_land[i+j] = float(
            np.sqrt(se_w_mask.sum() / denom)
        )
    i += bsz

# --------------------------
# PPE baseline RMSEs
# --------------------------
ppe_rmse_per_sample = np.zeros(N, dtype=np.float32)
ppe_rmse_per_sample_land = np.zeros(N, dtype=np.float32)

for n in range(N):

    pred_dP  = ppe_mean_dP                 # (1,1,H,W)
    true_dP  = y[n:n+1]                    # (1,1,H,W)

    diff = pred_dP - true_dP
    se   = diff * diff

    # area-weighted global RMSE
    se_w = se * lat_weights[None,None,:,None]
    ppe_rmse_per_sample[n] = float(np.sqrt(se_w.mean()))

    # land-only RMSE
    mask = landmask[None,None,:,:]
    se_w_mask = se_w * mask
    denom = float((landmask * lat_weights[:,None]).sum() + 1e-12)
    ppe_rmse_per_sample_land[n] = float(np.sqrt(se_w_mask.sum() / denom))


# --------------------------
# Print summary
# --------------------------
file_ids = np.arange(N)
print("\n" + "="*60)
print("RESULTS (All Members):")
print("="*60)
for name, idx in [("Train", train_ind), ("Val", val_ind), ("Test", test_ind)]:
    print(f"\n{name} Set (Global):")
    print(f"  PPE Mean RMSE:       {ppe_rmse_per_sample[idx].mean():.4f}")
    print(f"  Softmax Ens RMSE:    {nn_ens_rmse_per_sample[idx].mean():.4f}")
    print(f"  Members RMSE:        {member_rmse_per_sample[:, idx].mean():.4f} ± {member_rmse_per_sample[:, idx].std():.4f}")
    imp = (1 - nn_ens_rmse_per_sample[idx].mean() / (ppe_rmse_per_sample[idx].mean()+1e-12)) * 100
    print(f"  Improvement:         {imp:.2f}%")

    print(f"\n{name} Set (Land Only):")
    print(f"  PPE Mean RMSE:       {ppe_rmse_per_sample_land[idx].mean():.4f}")
    print(f"  Softmax Ens RMSE:    {nn_ens_rmse_per_sample_land[idx].mean():.4f}")
    print(f"  Members RMSE:        {member_rmse_per_sample_land[:, idx].mean():.4f} ± {member_rmse_per_sample_land[:, idx].std():.4f}")
    impL = (1 - nn_ens_rmse_per_sample_land[idx].mean() / (ppe_rmse_per_sample_land[idx].mean()+1e-12)) * 100
    print(f"  Improvement:         {impL:.2f}%")


# --------------------------
# Save compact arrays
# --------------------------
results = {
    "file_ids": file_ids.tolist(),
    "rmse_ppe": ppe_rmse_per_sample.tolist(),
    "rmse_softmax_mean": nn_ens_rmse_per_sample.tolist(),
    "rmse_softmax_members": member_rmse_per_sample.tolist(),
    "rmse_ppe_land": ppe_rmse_per_sample_land.tolist(),
    "rmse_softmax_mean_land": nn_ens_rmse_per_sample_land.tolist(),
    "rmse_softmax_members_land": member_rmse_per_sample_land.tolist(),
    "n_ensemble_members": int(n_ens),
    "train_indices": train_ind.tolist(),
    "val_indices": val_ind.tolist(),
    "test_indices": test_ind.tolist(),
    "gn_groups": int(gn_groups),
    "units": "mm/yr (PR_future space)",
}
with open(ens_dir / "softmax_ensemble_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

np.savez_compressed(
    ens_dir / "softmax_ensemble_analysis_arrays.npz",
    file_ids=file_ids,
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

print("\nAnalysis complete. " \
"Variables available for plotting: avg_probs, " \
"X (PR_hist), y (dP), X_norm, bin_centers," \
" train_ind/val_ind/test_ind, landmask, " \
"ds_input.latitude/longitude," \
" top5_mu_mean, member_ranking.")