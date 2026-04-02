r"""
Post-hoc calibration for softmax ProbUNet outputs.

This workflow compares four variants:
  1. Raw ensemble probabilities
  2. Temperature scaling only
  3. Affine bin-center recalibration
  4. Affine bin-center recalibration + temperature scaling

The affine-center model recalibrates the normalized target axis directly:

    c'_i = a + b * c_i

where:
  - c_i is the original normalized bin center
  - a is a location correction
  - b is a spread correction

Temperature scaling is then optionally applied on top of the probabilities:

    q_i \propto p_i^(1 / T)

Fitting is done on the validation split only, using a PIT-uniform objective
(Cramer-von Mises distance) rather than MACE. MACE is still reported.
"""

import glob
import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import xarray as xr
from tqdm import tqdm

from unet import ProbUNet


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
NUM_VAL_SAMPLES = None
NUM_TEST_SAMPLES = None
BATCH_SIZE = 20
RNG_SEED = 42

lat_dim = 128
num_bins = 64
base_channels = 200
gn_groups = 1
kernel_size = 3

dP_min = -700
dP_max = 1200

# Fit on a subset of validation land pixels for speed.
FIT_LAND_POINTS = 30000

TEMP_GRID = np.unique(
    np.concatenate(
        [
            np.array([0.75, 0.9, 1.0], dtype=np.float32),
            np.linspace(1.1, 4.0, 16, dtype=np.float32),
        ]
    )
)
SHIFT_GRID = np.linspace(-0.75, 0.75, 25, dtype=np.float32)
SCALE_GRID = np.linspace(0.70, 1.30, 13, dtype=np.float32)
EXPECTED_PROBS = np.linspace(0.0, 1.0, 21, dtype=np.float32)

base_dir = Path("/Users/ewellmeyer/Documents/research")
data_dir = base_dir / "HadGEM"
weights_dir = base_dir / "weights"

input_file = data_dir / f"GA789_PR_his_rg{lat_dim}.nc"
truth_file = data_dir / f"GA789_dPdK_rg{lat_dim}.nc"
landmask_file = data_dir / "hadgem_landmask_rg128.nc"

ens_name = (
    f"unet_ens_HG789_PR_dPdK_Softmax_unet6R_ch{base_channels}_k{kernel_size}_"
    f"{lat_dim}x_dPbins{num_bins}_gn{gn_groups}_dpmin{dP_min}_dPmax{dP_max}"
)
ens_dir = weights_dir / ens_name

norm_stats_path = ens_dir / "norm_stats.json"
bin_info_path = ens_dir / "born_bins.json"
split_ind_path = ens_dir / "data_splits.npz"
calibration_out_path = ens_dir / "temperature_calibration.json"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
def maybe_subsample(indices, limit, seed_offset=0):
    if limit is None or limit >= len(indices):
        return np.asarray(indices)
    rng = np.random.default_rng(RNG_SEED + seed_offset)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def load_data():
    with open(norm_stats_path, "r") as f:
        stats = json.load(f)
    x_mean = np.array(stats["x_mean"], dtype=np.float32).reshape(1, 1, 1, 1)
    x_std = np.array(stats["x_std"], dtype=np.float32).reshape(1, 1, 1, 1)
    y_mean = float(stats["y_mean"])
    y_std = float(stats["y_std"])

    with open(bin_info_path, "r") as f:
        bin_info = json.load(f)
    bin_centers = np.array(bin_info["bin_centers_norm"], dtype=np.float32)

    splits = np.load(split_ind_path)
    val_indices = maybe_subsample(splits["val"], NUM_VAL_SAMPLES, seed_offset=1)
    test_indices = maybe_subsample(splits["test"], NUM_TEST_SAMPLES, seed_offset=2)

    ds_in = xr.open_dataset(input_file)
    ds_tar = xr.open_dataset(truth_file)
    ds_lm = xr.open_dataset(landmask_file)

    X_full = ds_in.to_array().values.astype(np.float32)
    y_full = ds_tar.to_array().values.astype(np.float32)
    X_full = np.transpose(X_full, (1, 0, 2, 3))
    y_full = np.transpose(y_full, (1, 0, 2, 3))
    landmask = ds_lm["land_mask"].values.astype(bool)

    ds_in.close()
    ds_tar.close()
    ds_lm.close()

    X_full_norm = (X_full - x_mean) / x_std
    y_full_norm = (y_full - y_mean) / y_std

    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples:       {len(test_indices)}")

    return {
        "X_val": X_full_norm[val_indices],
        "y_val": y_full_norm[val_indices],
        "X_test": X_full_norm[test_indices],
        "y_test": y_full_norm[test_indices],
        "bin_centers": bin_centers,
        "landmask": landmask,
        "y_mean": y_mean,
        "y_std": y_std,
    }


# ==============================================================================
# 3. ENSEMBLE INFERENCE
# ==============================================================================
def get_member_files():
    member_files = sorted(glob.glob(str(ens_dir / f"{ens_dir.name}_member*.pth")))
    if not member_files:
        raise RuntimeError(f"No checkpoint files found in {ens_dir}")
    print(f"Found {len(member_files)} ensemble members")
    return member_files


def make_prob_memmap(split_name, X_norm, member_files):
    N, _, H, W = X_norm.shape
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"{split_name}_avg_probs_",
        suffix=".dat",
        dir="/tmp",
        delete=False,
    )
    tmp.close()

    avg_probs = np.memmap(tmp.name, dtype=np.float32, mode="w+", shape=(N, num_bins, H, W))
    avg_probs[:] = 0.0

    model = ProbUNet(1, base_channels, kernel_size, 0.0, num_bins, gn_groups=gn_groups).to(device)

    print(f"Accumulating ensemble probabilities for {split_name}...")
    for m_file in tqdm(member_files, desc=f"{split_name} members"):
        ckpt = torch.load(m_file, map_location=device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        model.eval()

        with torch.inference_mode():
            for i in range(0, N, BATCH_SIZE):
                xb = torch.as_tensor(X_norm[i : i + BATCH_SIZE], dtype=torch.float32, device=device)
                probs = model.forward_components(xb).float().cpu().numpy()
                avg_probs[i : i + probs.shape[0]] += probs / len(member_files)

        avg_probs.flush()

    return tmp.name, (N, num_bins, H, W)


# ==============================================================================
# 4. CALIBRATION HELPERS
# ==============================================================================
def apply_temperature_points(point_probs, temperature):
    if np.isclose(float(temperature), 1.0):
        return point_probs

    log_probs = np.log(np.clip(point_probs, 1e-12, 1.0)).astype(np.float32)
    scaled = log_probs / np.float32(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    scaled = np.exp(scaled)
    scaled /= scaled.sum(axis=1, keepdims=True)
    return scaled.astype(np.float32)


def apply_temperature_map(probs, temperature):
    if np.isclose(float(temperature), 1.0):
        return probs

    log_probs = np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32)
    scaled = log_probs / np.float32(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    scaled = np.exp(scaled)
    scaled /= scaled.sum(axis=1, keepdims=True)
    return scaled.astype(np.float32)


def transformed_centers(bin_centers, shift, scale):
    return np.float32(shift) + np.float32(scale) * bin_centers


def pit_from_point_probs(point_probs, y_points, centers):
    cdf = np.cumsum(point_probs, axis=1)
    cdf[:, -1] = 1.0

    idx = np.searchsorted(centers, y_points)
    idx = np.clip(idx, 1, len(centers) - 1)

    x0 = centers[idx - 1]
    x1 = centers[idx]
    y0 = cdf[np.arange(len(y_points)), idx - 1]
    y1 = cdf[np.arange(len(y_points)), idx]
    pit = y0 + (y_points - x0) * (y1 - y0) / np.maximum(x1 - x0, 1e-12)
    return np.clip(pit, 0.0, 1.0)


def pit_from_map_probs(probs, y_true_norm, centers, landmask):
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1, :, :] = 1.0

    y_flat = y_true_norm[:, 0][:, landmask].reshape(-1)
    cdf_flat = np.transpose(cdf, (0, 2, 3, 1))[:, landmask, :].reshape(-1, num_bins)

    idx = np.searchsorted(centers, y_flat)
    idx = np.clip(idx, 1, len(centers) - 1)

    x0 = centers[idx - 1]
    x1 = centers[idx]
    y0 = cdf_flat[np.arange(len(y_flat)), idx - 1]
    y1 = cdf_flat[np.arange(len(y_flat)), idx]
    pit = y0 + (y_flat - x0) * (y1 - y0) / np.maximum(x1 - x0, 1e-12)
    return np.clip(pit, 0.0, 1.0)


def mace_from_pit(pit_values):
    counts = np.zeros(len(EXPECTED_PROBS), dtype=np.int64)
    for i, p in enumerate(EXPECTED_PROBS):
        lower_q = 0.5 - p / 2.0
        upper_q = 0.5 + p / 2.0
        counts[i] = np.sum((pit_values >= lower_q) & (pit_values <= upper_q))
    observed = counts / max(len(pit_values), 1)
    mace = np.mean(np.abs(observed - EXPECTED_PROBS))
    return float(mace), observed


def cvm_from_pit(pit_values):
    if len(pit_values) == 0:
        return float("inf")
    u = np.sort(np.asarray(pit_values, dtype=np.float64))
    n = len(u)
    grid = (2.0 * np.arange(1, n + 1) - 1.0) / (2.0 * n)
    return float((1.0 / (12.0 * n)) + np.sum((u - grid) ** 2))


def point_mean_stats(point_probs, y_points, centers):
    mu = (point_probs * centers[None, :]).sum(axis=1)
    diff = mu - y_points
    return {
        "point_rmse_norm": float(np.sqrt(np.mean(diff ** 2))),
        "point_bias_norm": float(np.mean(diff)),
    }


def select_best(rows, primary_key="cvm", tie_key="point_rmse_norm", rel_tol=0.02):
    best_primary = min(row[primary_key] for row in rows)
    cutoff = best_primary * (1.0 + rel_tol) + 1e-12
    candidates = [row for row in rows if row[primary_key] <= cutoff]
    return min(candidates, key=lambda row: (row[tie_key], abs(row["mean_pit"] - 0.5)))


# ==============================================================================
# 5. FIT ON VALIDATION SUBSET
# ==============================================================================
def sample_val_points(prob_path, prob_shape, y_val_norm, landmask, max_points):
    prob_mm = np.memmap(prob_path, dtype=np.float32, mode="r", shape=prob_shape)

    land_i, land_j = np.where(landmask)
    n_land = len(land_i)
    total_points = prob_shape[0] * n_land
    n_take = min(max_points, total_points)

    rng = np.random.default_rng(RNG_SEED)
    flat_idx = np.sort(rng.choice(total_points, size=n_take, replace=False))
    sample_idx = flat_idx // n_land
    land_idx = flat_idx % n_land
    ii = land_i[land_idx]
    jj = land_j[land_idx]

    point_probs = np.asarray(prob_mm[sample_idx, :, ii, jj], dtype=np.float32)
    y_points = np.asarray(y_val_norm[sample_idx, 0, ii, jj], dtype=np.float32)
    return point_probs, y_points


def fit_temp_only(point_probs, y_points, bin_centers):
    rows = []
    print("Fitting temperature-only calibration on validation subset...")
    for temperature in TEMP_GRID:
        probs_t = apply_temperature_points(point_probs, temperature)
        pit = pit_from_point_probs(probs_t, y_points, bin_centers)
        stats = point_mean_stats(probs_t, y_points, bin_centers)
        rows.append(
            {
                "temperature": float(temperature),
                "shift": 0.0,
                "scale": 1.0,
                "cvm": cvm_from_pit(pit),
                "mace": mace_from_pit(pit)[0],
                "mean_pit": float(pit.mean()),
                **stats,
            }
        )
    return select_best(rows), rows


def fit_affine_given_temperature(point_probs, y_points, bin_centers, temperature):
    rows = []
    probs_t = apply_temperature_points(point_probs, temperature)
    print(f"Fitting affine-center calibration at fixed T={temperature:.3f}...")
    for shift in SHIFT_GRID:
        for scale in SCALE_GRID:
            centers = transformed_centers(bin_centers, shift, scale)
            pit = pit_from_point_probs(probs_t, y_points, centers)
            stats = point_mean_stats(probs_t, y_points, centers)
            rows.append(
                {
                    "temperature": float(temperature),
                    "shift": float(shift),
                    "scale": float(scale),
                    "cvm": cvm_from_pit(pit),
                    "mace": mace_from_pit(pit)[0],
                    "mean_pit": float(pit.mean()),
                    **stats,
                }
            )
    return select_best(rows), rows


def fit_affine_then_temp(point_probs, y_points, bin_centers):
    affine_1, affine_rows_1 = fit_affine_given_temperature(point_probs, y_points, bin_centers, 1.0)

    temp_rows_1 = []
    print("Refining temperature with affine centers fixed...")
    centers = transformed_centers(bin_centers, affine_1["shift"], affine_1["scale"])
    for temperature in TEMP_GRID:
        probs_t = apply_temperature_points(point_probs, temperature)
        pit = pit_from_point_probs(probs_t, y_points, centers)
        stats = point_mean_stats(probs_t, y_points, centers)
        temp_rows_1.append(
            {
                "temperature": float(temperature),
                "shift": float(affine_1["shift"]),
                "scale": float(affine_1["scale"]),
                "cvm": cvm_from_pit(pit),
                "mace": mace_from_pit(pit)[0],
                "mean_pit": float(pit.mean()),
                **stats,
            }
        )
    temp_1 = select_best(temp_rows_1)

    affine_2, affine_rows_2 = fit_affine_given_temperature(point_probs, y_points, bin_centers, temp_1["temperature"])

    temp_rows_2 = []
    print("Final temperature refinement...")
    centers = transformed_centers(bin_centers, affine_2["shift"], affine_2["scale"])
    for temperature in TEMP_GRID:
        probs_t = apply_temperature_points(point_probs, temperature)
        pit = pit_from_point_probs(probs_t, y_points, centers)
        stats = point_mean_stats(probs_t, y_points, centers)
        temp_rows_2.append(
            {
                "temperature": float(temperature),
                "shift": float(affine_2["shift"]),
                "scale": float(affine_2["scale"]),
                "cvm": cvm_from_pit(pit),
                "mace": mace_from_pit(pit)[0],
                "mean_pit": float(pit.mean()),
                **stats,
            }
        )
    final = select_best(temp_rows_2)

    return affine_1, final, {
        "affine_pass_1": affine_rows_1,
        "temp_pass_1": temp_rows_1,
        "affine_pass_2": affine_rows_2,
        "temp_pass_2": temp_rows_2,
    }


# ==============================================================================
# 6. FULL TEST EVALUATION
# ==============================================================================
def evaluate_variant(prob_path, prob_shape, y_true_norm, bin_centers, landmask, y_mean, y_std, temperature, shift, scale):
    prob_mm = np.memmap(prob_path, dtype=np.float32, mode="r", shape=prob_shape)
    centers = transformed_centers(bin_centers, shift, scale)

    pits = []
    global_sse = 0.0
    land_sse = 0.0
    global_n = 0
    land_n = 0
    sum_error = 0.0
    sum_land_error = 0.0

    for i in tqdm(
        range(0, prob_shape[0], BATCH_SIZE),
        desc=f"eval T={temperature:.2f}, a={shift:.3f}, b={scale:.3f}",
    ):
        probs = np.asarray(prob_mm[i : i + BATCH_SIZE])
        probs_t = apply_temperature_map(probs, temperature)
        y_batch = y_true_norm[i : i + BATCH_SIZE]

        pit_batch = pit_from_map_probs(probs_t, y_batch, centers, landmask)
        pits.append(pit_batch)

        mu_norm = (probs_t * centers[None, :, None, None]).sum(axis=1)
        truth_norm = y_batch[:, 0]
        mu = mu_norm * y_std + y_mean
        truth = truth_norm * y_std + y_mean
        diff = mu - truth

        global_sse += float((diff ** 2).sum())
        land_sse += float(((diff ** 2) * landmask[None]).sum())
        global_n += diff.size
        land_n += diff.shape[0] * int(landmask.sum())
        sum_error += float(diff.sum())
        sum_land_error += float((diff * landmask[None]).sum())

    pits = np.concatenate(pits)
    mace, observed = mace_from_pit(pits)

    return {
        "temperature": float(temperature),
        "shift": float(shift),
        "scale": float(scale),
        "pit": pits,
        "mace": mace,
        "observed": observed,
        "mean_pit": float(pits.mean()),
        "global_rmse": float(np.sqrt(global_sse / max(global_n, 1))),
        "land_rmse": float(np.sqrt(land_sse / max(land_n, 1))),
        "mean_error": float(sum_error / max(global_n, 1)),
        "land_mean_error": float(sum_land_error / max(land_n, 1)),
    }


# ==============================================================================
# 7. PLOTTING
# ==============================================================================
def rows_to_heatmap(rows, value_key):
    grid = np.full((len(SCALE_GRID), len(SHIFT_GRID)), np.nan, dtype=np.float32)
    shift_to_idx = {float(v): i for i, v in enumerate(SHIFT_GRID)}
    scale_to_idx = {float(v): i for i, v in enumerate(SCALE_GRID)}
    for row in rows:
        i = scale_to_idx[float(row["scale"])]
        j = shift_to_idx[float(row["shift"])]
        grid[i, j] = row[value_key]
    return grid


def plot_fit_diagnostics(temp_rows, affine_rows, affine_temp_rows):
    sns.set_context("paper", font_scale=1.15)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].plot(
        [row["temperature"] for row in temp_rows],
        [row["cvm"] for row in temp_rows],
        marker="o",
        color="tab:blue",
    )
    ax[0].set_title("Temperature-Only Fit", fontweight="bold")
    ax[0].set_xlabel("Temperature")
    ax[0].set_ylabel("Validation CvM")
    ax[0].grid(True, linestyle=":", alpha=0.5)

    hm = rows_to_heatmap(affine_rows, "cvm")
    sns.heatmap(
        hm,
        ax=ax[1],
        cmap="viridis",
        cbar=True,
        xticklabels=[f"{v:.2f}" for v in SHIFT_GRID],
        yticklabels=[f"{v:.2f}" for v in SCALE_GRID],
    )
    ax[1].set_title("Affine Fit at T=1", fontweight="bold")
    ax[1].set_xlabel("Shift a")
    ax[1].set_ylabel("Scale b")

    hm = rows_to_heatmap(affine_temp_rows, "cvm")
    sns.heatmap(
        hm,
        ax=ax[2],
        cmap="viridis",
        cbar=True,
        xticklabels=[f"{v:.2f}" for v in SHIFT_GRID],
        yticklabels=[f"{v:.2f}" for v in SCALE_GRID],
    )
    ax[2].set_title("Affine Fit at Final T", fontweight="bold")
    ax[2].set_xlabel("Shift a")
    ax[2].set_ylabel("Scale b")

    plt.tight_layout()
    plt.show()


def plot_reliability_quad(raw, temp_only, affine, affine_temp):
    sns.set_context("paper", font_scale=1.1)
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))

    variants = [
        ("Raw", raw, "steelblue"),
        ("Temp Only", temp_only, "seagreen"),
        ("Affine", affine, "mediumpurple"),
        ("Affine + Temp", affine_temp, "darkorange"),
    ]

    for col, (title, result, color) in enumerate(variants):
        sns.histplot(result["pit"], bins=20, stat="density", ax=ax[0, col], color=color, edgecolor="black", alpha=0.7)
        ax[0, col].axhline(1.0, color="red", linestyle="--", linewidth=2)
        ax[0, col].set_title(
            f"{title}\nMACE={result['mace']:.4f}, mean PIT={result['mean_pit']:.3f}",
            fontweight="bold",
        )
        ax[0, col].set_xlabel("Cumulative Probability at Truth")
        ax[0, col].set_ylabel("Density")

        ax[1, col].plot(EXPECTED_PROBS, result["observed"], "o-", color=color, linewidth=2)
        ax[1, col].plot([0, 1], [0, 1], "k--", linewidth=2)
        ax[1, col].set_title(
            f"T={result['temperature']:.3f}, a={result['shift']:.3f}, b={result['scale']:.3f}",
            fontweight="bold",
        )
        ax[1, col].set_xlabel("Predicted Confidence Interval")
        ax[1, col].set_ylabel("Observed Frequency")
        ax[1, col].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 8. MAIN EXECUTION
# ==============================================================================
data = load_data()
member_files = get_member_files()
val_prob_path = None
test_prob_path = None

try:
    val_prob_path, val_prob_shape = make_prob_memmap("val", data["X_val"], member_files)
    point_probs, y_points = sample_val_points(
        val_prob_path,
        val_prob_shape,
        data["y_val"],
        data["landmask"],
        FIT_LAND_POINTS,
    )

    temp_only_best, temp_rows = fit_temp_only(point_probs, y_points, data["bin_centers"])
    affine_best, affine_temp_best, affine_history = fit_affine_then_temp(point_probs, y_points, data["bin_centers"])

    print("\nValidation-fit summary:")
    print(
        f"  Temp only     | T={temp_only_best['temperature']:.4f}  "
        f"CvM={temp_only_best['cvm']:.4f}  MACE={temp_only_best['mace']:.4f}  "
        f"mean PIT={temp_only_best['mean_pit']:.4f}"
    )
    print(
        f"  Affine only   | a={affine_best['shift']:.4f}  b={affine_best['scale']:.4f}  "
        f"CvM={affine_best['cvm']:.4f}  MACE={affine_best['mace']:.4f}  "
        f"mean PIT={affine_best['mean_pit']:.4f}"
    )
    print(
        f"  Affine + Temp | T={affine_temp_best['temperature']:.4f}  "
        f"a={affine_temp_best['shift']:.4f}  b={affine_temp_best['scale']:.4f}  "
        f"CvM={affine_temp_best['cvm']:.4f}  MACE={affine_temp_best['mace']:.4f}  "
        f"mean PIT={affine_temp_best['mean_pit']:.4f}"
    )

    plot_fit_diagnostics(
        temp_rows,
        affine_history["affine_pass_1"],
        affine_history["affine_pass_2"],
    )

    test_prob_path, test_prob_shape = make_prob_memmap("test", data["X_test"], member_files)

    raw_result = evaluate_variant(
        test_prob_path, test_prob_shape, data["y_test"], data["bin_centers"],
        data["landmask"], data["y_mean"], data["y_std"], 1.0, 0.0, 1.0
    )
    temp_result = evaluate_variant(
        test_prob_path, test_prob_shape, data["y_test"], data["bin_centers"],
        data["landmask"], data["y_mean"], data["y_std"], temp_only_best["temperature"], 0.0, 1.0
    )
    affine_result = evaluate_variant(
        test_prob_path, test_prob_shape, data["y_test"], data["bin_centers"],
        data["landmask"], data["y_mean"], data["y_std"], 1.0, affine_best["shift"], affine_best["scale"]
    )
    affine_temp_result = evaluate_variant(
        test_prob_path, test_prob_shape, data["y_test"], data["bin_centers"],
        data["landmask"], data["y_mean"], data["y_std"],
        affine_temp_best["temperature"], affine_temp_best["shift"], affine_temp_best["scale"]
    )

    print("\nTest split summary:")
    for label, result in [
        ("Raw", raw_result),
        ("Temp only", temp_result),
        ("Affine", affine_result),
        ("Affine + Temp", affine_temp_result),
    ]:
        print(
            f"  {label:13s} | MACE={result['mace']:.4f}  "
            f"mean PIT={result['mean_pit']:.4f}  "
            f"global RMSE={result['global_rmse']:.4f}  "
            f"land RMSE={result['land_rmse']:.4f}  "
            f"mean error={result['mean_error']:.4f}"
        )

    plot_reliability_quad(raw_result, temp_result, affine_result, affine_temp_result)

    with open(calibration_out_path, "w") as f:
        json.dump(
            {
                "fit_land_points": int(len(y_points)),
                "temperature_grid": TEMP_GRID.tolist(),
                "shift_grid": SHIFT_GRID.tolist(),
                "scale_grid": SCALE_GRID.tolist(),
                "validation_temp_only_best": temp_only_best,
                "validation_affine_best": affine_best,
                "validation_affine_temp_best": affine_temp_best,
                "test_raw": {
                    "mace": raw_result["mace"],
                    "mean_pit": raw_result["mean_pit"],
                    "global_rmse": raw_result["global_rmse"],
                    "land_rmse": raw_result["land_rmse"],
                    "mean_error": raw_result["mean_error"],
                    "land_mean_error": raw_result["land_mean_error"],
                },
                "test_temp_only": {
                    "temperature": temp_result["temperature"],
                    "mace": temp_result["mace"],
                    "mean_pit": temp_result["mean_pit"],
                    "global_rmse": temp_result["global_rmse"],
                    "land_rmse": temp_result["land_rmse"],
                    "mean_error": temp_result["mean_error"],
                    "land_mean_error": temp_result["land_mean_error"],
                },
                "test_affine": {
                    "shift": affine_result["shift"],
                    "scale": affine_result["scale"],
                    "mace": affine_result["mace"],
                    "mean_pit": affine_result["mean_pit"],
                    "global_rmse": affine_result["global_rmse"],
                    "land_rmse": affine_result["land_rmse"],
                    "mean_error": affine_result["mean_error"],
                    "land_mean_error": affine_result["land_mean_error"],
                },
                "test_affine_temp": {
                    "temperature": affine_temp_result["temperature"],
                    "shift": affine_temp_result["shift"],
                    "scale": affine_temp_result["scale"],
                    "mace": affine_temp_result["mace"],
                    "mean_pit": affine_temp_result["mean_pit"],
                    "global_rmse": affine_temp_result["global_rmse"],
                    "land_rmse": affine_temp_result["land_rmse"],
                    "mean_error": affine_temp_result["mean_error"],
                    "land_mean_error": affine_temp_result["land_mean_error"],
                },
            },
            f,
            indent=2,
        )

    print(f"\nSaved calibration summary to {calibration_out_path}")
finally:
    if val_prob_path and os.path.exists(val_prob_path):
        os.remove(val_prob_path)
    if test_prob_path and os.path.exists(test_prob_path):
        os.remove(test_prob_path)
