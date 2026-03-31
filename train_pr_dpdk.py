"""
Softmax-style probabilistic regression with a U-Net backbone.
- Bins defined uniformly in target space (dP-like), then normalized with HadGEM train stats
- Per-bin σ from local spacing (soft targets)
- TRAIN: HadGEM PPE (PR -> dPdP)
- VAL:   HadGEM internal val split
"""

import os, json, gc, random
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# --------------------------
# Device
# --------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
use_amp = torch.backends.mps.is_available()
print(f"Using device: {device} | AMP: {use_amp}")

# --------------------------
# Config
# --------------------------
ensemble_size = 10
base_seed     = 42
base_ch       = 8
gn_groups     = 1
k_size        = 3
pdrop         = 0.0
num_bins      = 64
sigma_scale   = 0.6
batch_train   = 40
batch_val     = 40
num_epochs    = 5000
patience      = 20
grad_clip     = 1.0

# target bin range (physical target space; will be normalized using HadGEM train stats)
dP_min   = -700
dP_max   =  1200

random.seed(base_seed); np.random.seed(base_seed); torch.manual_seed(base_seed)

# --------------------------
# Paths
# --------------------------
base_model_name = (
    f"ens_HG789_PR_dPdK_Softmax_unet6R_ch{base_ch}_k{k_size}_"
    f"128x_dPbins{num_bins}_gn{gn_groups}_dpmin{dP_min}_dPmax{dP_max}"
)
weights_dir  = "/Users/ewellmeyer/Documents/research/weights"
ensemble_dir = os.path.join(weights_dir, base_model_name)
os.makedirs(ensemble_dir, exist_ok=True)

# HadGEM train data
hadgem_dir  = "/Users/ewellmeyer/Documents/research/HadGEM"
input_file  = os.path.join(hadgem_dir, "GA789_PR_his_rg128.nc")
truth_file  = os.path.join(hadgem_dir, "GA789_dPdK_rg128.nc")

# --------------------------
# Dataset
# --------------------------
class ClimateDataset(torch.utils.data.Dataset):
    def __init__(self, X, Ydist):
        # X: (N,H,W,C), Ydist: (N,H,W,B)
        X = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)   # -> (N,C,H,W)
        Y = np.transpose(Ydist, (0, 3, 1, 2)).astype(np.float32) # -> (N,B,H,W)
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

# --------------------------
# Model
# --------------------------

class CustomPad(nn.Module):
    def __init__(self, pad_height, pad_width):
        """
        Pads the input tensor with reflect padding for the height dimension
        and circular padding for the width dimension.
        """
        super(CustomPad, self).__init__()
        self.pad_height = pad_height
        self.pad_width = pad_width

    def forward(self, x):
        x = F.pad(x, (0, 0, self.pad_height, self.pad_height), mode='reflect')
        x = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode='circular')
        return x
    
class ConvResBlockSingle(nn.Module):
    """
    A simplified residual block with only one convolutional layer.
    This version is computationally faster but may have less representational power
    than a block with two convolutional layers.
    """
    def __init__(self, in_ch, out_ch, k_size=3, p_drop=0.0, gn_groups: int = 1):  
        super().__init__()
        pad = (k_size - 1) // 2
        self.pad = CustomPad(pad, pad)

        # A single convolution layer
        self.conv1 = nn.Conv2d(in_ch, out_ch, k_size, padding=0)

        # Normalization for the single convolution layer
        self.gn1 = nn.GroupNorm(num_groups=gn_groups, num_channels=out_ch)  # <- use gn_groups
        # self.gn1 = nn.GroupNorm(num_groups=out_ch, num_channels=out_ch)

        self.act = nn.Mish(inplace=True)
        self.dp = nn.Dropout2d(p_drop) if p_drop else nn.Identity()
        
        # Skip connection to match input and output channels for the residual addition
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        """
        Defines the forward pass.
        The path is now: CONV -> NORM -> ACT -> DROPOUT
        """
        # Main path with a single convolutional step
        y = self.act(self.gn1(self.conv1(self.pad(x))))
        y = self.dp(y)
        
        # Add the skip connection (input) to the output of the main path
        return self.act(y + self.skip(x))

class Unet6R(nn.Module):
    """
    A U-Net architecture with 6 levels of encoding and decoding.
    The architecture is designed to handle input sizes where
    the dimensions are not necessarily divisible by 2^6 = 64
    by automatically padding and cropping internally.
    """
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 32,
        kernel_size: int = 3,
        p_drop: float = 0.1,
        gn_groups: int = 1,
    ):
        super().__init__()
        k = kernel_size
        c1, c2, c4, c8, c16, c32, c64 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
            base_channels * 32,
            base_channels * 64,
        )

        # ---- Encoder ----
        self.enc1 = ConvResBlockSingle(input_channels, c1, k_size=k, p_drop=p_drop)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvResBlockSingle(c1, c2, k_size=k, p_drop=p_drop)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvResBlockSingle(c2, c4, k_size=k, p_drop=p_drop)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvResBlockSingle(c4, c8, k_size=k, p_drop=p_drop)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5 = ConvResBlockSingle(c8, c16, k_size=k, p_drop=p_drop)
        self.pool5 = nn.MaxPool2d(2)
        
        self.enc6 = ConvResBlockSingle(c16, c32, k_size=k, p_drop=p_drop)
        self.pool6 = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.bottleneck = ConvResBlockSingle(c32, c64, k_size=k, p_drop=p_drop)

        # ---- Decoder ----
        self.upconv1 = nn.ConvTranspose2d(c64, c32, kernel_size=2, stride=2)
        self.dec1 = ConvResBlockSingle(c32 + c32, c32, k_size=k, p_drop=p_drop)

        self.upconv2 = nn.ConvTranspose2d(c32, c16, kernel_size=2, stride=2)
        self.dec2 = ConvResBlockSingle(c16 + c16, c16, k_size=k, p_drop=p_drop)

        self.upconv3 = nn.ConvTranspose2d(c16, c8, kernel_size=2, stride=2)
        self.dec3 = ConvResBlockSingle(c8 + c8, c8, k_size=k, p_drop=p_drop)

        self.upconv4 = nn.ConvTranspose2d(c8, c4, kernel_size=2, stride=2)
        self.dec4 = ConvResBlockSingle(c4 + c4, c4, k_size=k, p_drop=p_drop)

        self.upconv5 = nn.ConvTranspose2d(c4, c2, kernel_size=2, stride=2)
        self.dec5 = ConvResBlockSingle(c2 + c2, c2, k_size=k, p_drop=p_drop)
        
        self.upconv6 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec6 = ConvResBlockSingle(c1 + c1, c1, k_size=k, p_drop=p_drop)

        # ---- Head ----
        self.final_conv = nn.Conv2d(c1, output_channels, kernel_size=1)

    def forward(self, x):
        # Store original size for final cropping
        original_h, original_w = x.shape[2], x.shape[3]

        # Calculate padding to make dimensions divisible by 2^6 = 64
        if original_h % 64 == 0 and original_w % 64 == 0:
            pad_h, pad_w = 0, 0
        else:
            pad_h = (64 - original_h % 64) % 64
            pad_w = (64 - original_w % 64) % 64

        # Pad the input tensor if necessary. The format is (left, right, top, bottom).
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='reflect')
        else:
            x_padded = x

        # Encoder Path using padded input
        x1 = self.enc1(x_padded)
        x1p = self.pool1(x1)
        x2 = self.enc2(x1p)
        x2p = self.pool2(x2)
        x3 = self.enc3(x2p)
        x3p = self.pool3(x3)
        x4 = self.enc4(x3p)
        x4p = self.pool4(x4)
        x5 = self.enc5(x4p)
        x5p = self.pool5(x5)
        x6 = self.enc6(x5p)
        x6p = self.pool6(x6)

        # Bottleneck
        b = self.bottleneck(x6p)

        # Decoder Path
        u1 = self.upconv1(b)
        d1 = self.dec1(torch.cat([u1, x6], dim=1))
        u2 = self.upconv2(d1)
        d2 = self.dec2(torch.cat([u2, x5], dim=1))
        u3 = self.upconv3(d2)
        d3 = self.dec3(torch.cat([u3, x4], dim=1))
        u4 = self.upconv4(d3)
        d4 = self.dec4(torch.cat([u4, x3], dim=1))
        u5 = self.upconv5(d4)
        d5 = self.dec5(torch.cat([u5, x2], dim=1))
        u6 = self.upconv6(d5)
        d6 = self.dec6(torch.cat([u6, x1], dim=1))

        output_padded = self.final_conv(d6)

        # Crop the output back to the original size if padding was added
        if pad_h > 0 or pad_w > 0:
            final_output = output_padded[:, :, pad_h // 2 : original_h + pad_h // 2, pad_w // 2 : original_w + pad_w // 2]
        else:
            final_output = output_padded

        return final_output

class SoftmaxHead(nn.Module):
    def __init__(self, in_channels, num_bins):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_bins, kernel_size=1)
    def forward(self, feat):
        logits = self.conv(feat)              # (N,B,H,W)
        probs  = torch.softmax(logits, dim=1) # (N,B,H,W)
        return probs

class ProbUNet(nn.Module):
    def __init__(self, input_channels, base_channels, kernel_size, p_drop, num_bins, gn_groups: int = 1):
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
    def forward(self, x):
        return self.head(self.backbone(x))

# --------------------------
# Helpers
# --------------------------
def make_soft_labels(y_norm, bin_centers_norm, sigma_bins_norm):
    """
    y_norm: (N,H,W,1) normalized target
    returns: (N,H,W,B) soft labels
    """
    # y_norm[...,0:1] - centers -> (N,H,W,B)
    diff = y_norm[..., 0:1] - bin_centers_norm.reshape(1, 1, 1, -1)
    Y = np.exp(-0.5 * (diff / (sigma_bins_norm.reshape(1, 1, 1, -1) + 1e-12))**2)
    Y = (Y / (Y.sum(axis=-1, keepdims=True) + 1e-12)).astype(np.float32)
    return Y

def nll_from_probs(probs, y_soft):
    # probs: (B,Bin,H,W)?? in torch: (N,B,H,W), y_soft same shape
    return -(y_soft * probs.clamp_min(1e-12).log()).sum(dim=1).mean()

# --------------------------
# Load HadGEM data
# --------------------------
print("Loading HadGEM data...")
ds_in  = xr.open_dataset(input_file)
ds_tgt = xr.open_dataset(truth_file)

X_raw = ds_in["PR"].values[..., np.newaxis].astype(np.float32)  # (N,H,W,1)
y_raw = ds_tgt["dPdP"].values[..., np.newaxis].astype(np.float32) # (N,H,W,1)

ds_in.close(); ds_tgt.close()

# --------------------------
# HadGEM splits
# --------------------------
idx = np.arange(len(X_raw))
tr_idx, tmp_idx = train_test_split(idx, test_size=0.3, random_state=base_seed)
va_idx, te_idx  = train_test_split(tmp_idx, test_size=0.2/0.3, random_state=base_seed)
np.savez(os.path.join(ensemble_dir, "data_splits.npz"), train=tr_idx, val=va_idx, test=te_idx)
print(f"HadGEM Splits: Train={len(tr_idx)}, Val={len(va_idx)}, Test={len(te_idx)}")

X_tr, y_tr = X_raw[tr_idx], y_raw[tr_idx]
X_va_hg, y_va_hg = X_raw[va_idx], y_raw[va_idx]

# Keep physical HG val target for RMSE
y_va_hg_tensor = torch.from_numpy(np.transpose(y_va_hg, (0, 3, 1, 2)).astype(np.float32))  # (N,1,H,W)

# Free big raw arrays once we slice
del X_raw, y_raw
gc.collect()

# --------------------------
# Normalization (stats from HadGEM TRAIN ONLY)
# --------------------------
Cx = X_tr.shape[-1]  # 1
x_mean = X_tr.reshape(-1, Cx).mean(axis=0)
x_std  = X_tr.reshape(-1, Cx).std(axis=0).clip(1e-6)

y_mean = float(y_tr.mean())
y_std  = float(y_tr.std().clip(1e-6))

print(f"HadGEM train stats: x={float(x_mean[0]):.4f}±{float(x_std[0]):.4f} | y={y_mean:.4f}±{y_std:.4f}")

X_tr_n    = (X_tr    - x_mean) / x_std
X_va_hg_n = (X_va_hg - x_mean) / x_std

y_tr_n    = (y_tr    - y_mean) / y_std
y_va_hg_n = (y_va_hg - y_mean) / y_std

with open(os.path.join(ensemble_dir, "norm_stats.json"), "w") as f:
    json.dump({"x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
               "y_mean": y_mean, "y_std": y_std}, f, indent=2)

# --------------------------
# Bin centers + per-bin sigma (in normalized target space)
# --------------------------
dP_centers = np.linspace(dP_min, dP_max, num_bins, dtype=np.float32)  # physical centers
bin_centers = ((dP_centers - y_mean) / y_std).astype(np.float32)      # normalized centers

diffs = np.diff(bin_centers)
spacing = np.r_[diffs[0], 0.5 * (diffs[:-1] + diffs[1:]), diffs[-1]]
sigma_bins = np.maximum(spacing * sigma_scale, 1e-4).astype(np.float32)

with open(os.path.join(ensemble_dir, "born_bins.json"), "w") as f:
    json.dump({
        "num_bins": int(num_bins),
        "bin_centers_norm": bin_centers.tolist(),
        "bin_centers_dP": dP_centers.tolist(),
        "sigma_bins_norm": sigma_bins.tolist(),
        "definition": "uniform_dP",
        "dP_min": float(dP_min),
        "dP_max": float(dP_max),
        "sigma_scale": float(sigma_scale)
    }, f, indent=2)

# --------------------------
# Precompute soft labels
# --------------------------
print("Building soft labels...")
Ytr    = make_soft_labels(y_tr_n,    bin_centers, sigma_bins)  # (Ntr,H,W,B)
Yva_hg = make_soft_labels(y_va_hg_n, bin_centers, sigma_bins)  # (Nhg,H,W,B)

# Save tensors for RMSE in physical space
bin_centers_t = torch.as_tensor(bin_centers, dtype=torch.float32, device=device).view(1,-1,1,1)

# --------------------------
# Build loaders
# --------------------------
train_ds = ClimateDataset(X_tr_n,    Ytr)
val_hg_ds= ClimateDataset(X_va_hg_n, Yva_hg)

train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True)
val_hg_loader= DataLoader(val_hg_ds, batch_size=batch_val,   shuffle=False)

# Free big arrays we don't need anymore
del X_tr, X_va_hg, y_tr, y_va_hg
del X_tr_n, X_va_hg_n, y_tr_n, y_va_hg_n
del Ytr, Yva_hg
gc.collect()

# --------------------------
# Training
# --------------------------
for member in range(ensemble_size):
    print(f"\n==== Training member {member} ====")

    final_path = os.path.join(ensemble_dir, f"{base_model_name}_member{member}.pth")
    best_path  = os.path.join(ensemble_dir, f"best_member{member}.pth")

    torch.manual_seed(base_seed + member)
    np.random.seed(base_seed + member)
    random.seed(base_seed + member)

    model = ProbUNet(1, base_ch, k_size, pdrop, num_bins, gn_groups=gn_groups).to(device)
    opt = optim.RAdam(model.parameters(), lr=1e-3)
    sch = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)

    # Resume if checkpoint exists
    best_val = float("inf")
    if os.path.exists(best_path):
        print(f"Resuming from {best_path}")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        best_val = ckpt.get("best_val_loss", best_val)

    epochs_bad = 0

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
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

        # ---- val (HadGEM split) ----
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

                # expected value -> physical
                mu_n = (probs * bin_centers_t).sum(dim=1, keepdim=True)  # normalized
                mu   = mu_n * y_std + y_mean

                y_true = y_va_hg_tensor[idx0:idx0+bs].to(device)
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

    # Save final
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    torch.save({"model": model.state_dict()}, final_path)
    print(f"Saved {final_path}")

print("\nTraining complete.")
