r"""
Post-hoc calibration for softmax ProbUNet outputs.

This workflow compares three variants:
  1. Raw ensemble probabilities
  2. Temperature scaling only
  3. Shift + temperature scaling

The shift term addresses PIT skew by tilting probability mass along the bin axis:

    q_i \propto exp((log p_i + beta * c_i) / T)

where:
  - p_i is the raw predicted probability for bin i
  - c_i is the normalized bin center
  - beta shifts the distribution left/right
  - T controls spread

Fitting is done on the validation split only.
Evaluation is reported on the test split.
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

# Fit parameters on a random subset of land pixels from the validation split.
# This keeps the search fast while evaluating the final metrics on all test points.
FIT_LAND_POINTS = 50_000

TEMP_GRID = np.unique(
    np.concatenate(
        [
            np.array([0.75, 0.9, 1.0], dtype=np.float32),
            np.linspace(1.1, 4.5, 18, dtype=np.float32),
        ]
    )
)
BETA_GRID = np.linspace(-0.25, 0.25, 21, dtype=np.float32)
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
def calibrate_probs_map(probs, temperature, beta, bin_centers):
    if np.isclose(float(temperature), 1.0) and np.isclose(float(beta), 0.0):
        return probs

    log_probs = np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32)
    shifted = log_probs + np.float32(beta) * bin_centers[None, :, None, None]
    shifted = shifted / np.float32(temperature)
    shifted -= shifted.max(axis=1, keepdims=True)
    shifted = np.exp(shifted)
    shifted /= shifted.sum(axis=1, keepdims=True)
    return shifted.astype(np.float32)


def calibrate_probs_points(point_probs, temperature, beta, bin_centers):
    if np.isclose(float(temperature), 1.0) and np.isclose(float(beta), 0.0):
        return point_probs

    log_probs = np.log(np.clip(point_probs, 1e-12, 1.0)).astype(np.float32)
    shifted = log_probs + np.float32(beta) * bin_centers[None, :]
    shifted = shifted / np.float32(temperature)
    shifted -= shifted.max(axis=1, keepdims=True)
    shifted = np.exp(shifted)
    shifted /= shifted.sum(axis=1, keepdims=True)
    return shifted.astype(np.float32)


def probs_to_pit_values(probs, y_true_norm, bin_centers, landmask):
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1, :, :] = 1.0

    y_flat = y_true_norm[:, 0][:, landmask].reshape(-1)
    cdf_flat = np.transpose(cdf, (0, 2, 3, 1))[:, landmask, :].reshape(-1, num_bins)

    idx = np.searchsorted(bin_centers, y_flat)
    idx = np.clip(idx, 1, num_bins - 1)

    x0 = bin_centers[idx - 1]
    x1 = bin_centers[idx]
    y0 = cdf_flat[np.arange(len(y_flat)), idx - 1]
    y1 = cdf_flat[np.arange(len(y_flat)), idx]

    pit = y0 + (y_flat - x0) * (y1 - y0) / np.maximum(x1 - x0, 1e-12)
    return np.clip(pit, 0.0, 1.0)


def point_probs_to_pit_values(point_probs, y_points, bin_centers):
    cdf = np.cumsum(point_probs, axis=1)
    cdf[:, -1] = 1.0

    idx = np.searchsorted(bin_centers, y_points)
    idx = np.clip(idx, 1, num_bins - 1)

    x0 = bin_centers[idx - 1]
    x1 = bin_centers[idx]
    y0 = cdf[np.arange(len(y_points)), idx - 1]
    y1 = cdf[np.arange(len(y_points)), idx]

    pit = y0 + (y_points - x0) * (y1 - y0) / np.maximum(x1 - x0, 1e-12)
    return np.clip(pit, 0.0, 1.0)


def interval_counts_from_pit(pit_values):
    counts = np.zeros(len(EXPECTED_PROBS), dtype=np.int64)
    for i, p in enumerate(EXPECTED_PROBS):
        lower_q = 0.5 - p / 2.0
        upper_q = 0.5 + p / 2.0
        counts[i] = np.sum((pit_values >= lower_q) & (pit_values <= upper_q))
    return counts


def mace_from_pit(pit_values):
    counts = interval_counts_from_pit(pit_values)
    observed = counts / max(len(pit_values), 1)
    mace = np.mean(np.abs(observed - EXPECTED_PROBS))
    return float(mace), observed


def mean_prediction_stats_from_probs(probs, y_true_norm, bin_centers, y_mean, y_std, landmask):
    mu_norm = (probs * bin_centers[None, :, None, None]).sum(axis=1)
    truth_norm = y_true_norm[:, 0]

    mu = mu_norm * y_std + y_mean
    truth = truth_norm * y_std + y_mean

    diff = mu - truth
    diff2 = diff ** 2

    return {
        "global_rmse": float(np.sqrt(diff2.mean())),
        "land_rmse": float(np.sqrt((diff2 * landmask[None]).sum() / max(int(landmask.sum()) * diff.shape[0], 1))),
        "mean_error": float(diff.mean()),
        "land_mean_error": float((diff * landmask[None]).sum() / max(int(landmask.sum()) * diff.shape[0], 1)),
    }


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


def fit_temperature_only(point_probs, y_points, bin_centers):
    best = None
    history = []

    print("Fitting temperature-only calibration on validation subset...")
    for temperature in TEMP_GRID:
        probs_t = calibrate_probs_points(point_probs, temperature, 0.0, bin_centers)
        pit = point_probs_to_pit_values(probs_t, y_points, bin_centers)
        mace, _ = mace_from_pit(pit)
        mean_pit = float(pit.mean())
        row = {"temperature": float(temperature), "mace": mace, "mean_pit": mean_pit}
        history.append(row)

        if best is None or mace < best["mace"]:
            best = row

    return best, history


def fit_shift_temperature(point_probs, y_points, bin_centers, temp_only_best):
    print("Fitting shift + temperature calibration on validation subset...")

    # Step 1: choose beta to center PIT at T = 1
    beta_center_hist = []
    best_beta_center = None
    for beta in BETA_GRID:
        probs_bt = calibrate_probs_points(point_probs, 1.0, beta, bin_centers)
        pit = point_probs_to_pit_values(probs_bt, y_points, bin_centers)
        mace, _ = mace_from_pit(pit)
        row = {
            "beta": float(beta),
            "mace": mace,
            "mean_pit": float(pit.mean()),
            "pit_bias": float(abs(pit.mean() - 0.5)),
        }
        beta_center_hist.append(row)
        if best_beta_center is None or row["pit_bias"] < best_beta_center["pit_bias"]:
            best_beta_center = row

    # Step 2: fit temperature with beta fixed
    temp_hist = []
    best_temp = None
    beta0 = best_beta_center["beta"]
    for temperature in TEMP_GRID:
        probs_bt = calibrate_probs_points(point_probs, temperature, beta0, bin_centers)
        pit = point_probs_to_pit_values(probs_bt, y_points, bin_centers)
        mace, _ = mace_from_pit(pit)
        row = {"temperature": float(temperature), "beta": float(beta0), "mace": mace, "mean_pit": float(pit.mean())}
        temp_hist.append(row)
        if best_temp is None or mace < best_temp["mace"]:
            best_temp = row

    # Step 3: refine beta with temperature fixed
    beta_refine_hist = []
    best_refined = None
    temp1 = best_temp["temperature"]
    for beta in BETA_GRID:
        probs_bt = calibrate_probs_points(point_probs, temp1, beta, bin_centers)
        pit = point_probs_to_pit_values(probs_bt, y_points, bin_centers)
        mace, _ = mace_from_pit(pit)
        row = {"temperature": float(temp1), "beta": float(beta), "mace": mace, "mean_pit": float(pit.mean())}
        beta_refine_hist.append(row)
        if best_refined is None or mace < best_refined["mace"]:
            best_refined = row

    # Step 4: final temperature refinement
    temp_refine_hist = []
    best_final = None
    beta1 = best_refined["beta"]
    for temperature in TEMP_GRID:
        probs_bt = calibrate_probs_points(point_probs, temperature, beta1, bin_centers)
        pit = point_probs_to_pit_values(probs_bt, y_points, bin_centers)
        mace, _ = mace_from_pit(pit)
        row = {"temperature": float(temperature), "beta": float(beta1), "mace": mace, "mean_pit": float(pit.mean())}
        temp_refine_hist.append(row)
        if best_final is None or mace < best_final["mace"]:
            best_final = row

    return best_final, {
        "beta_center": beta_center_hist,
        "temp_pass_1": temp_hist,
        "beta_refine": beta_refine_hist,
        "temp_refine": temp_refine_hist,
        "temp_only_best": temp_only_best,
    }


# ==============================================================================
# 6. FULL TEST EVALUATION
# ==============================================================================
def evaluate_variant(prob_path, prob_shape, y_true_norm, bin_centers, landmask, y_mean, y_std, temperature, beta):
    prob_mm = np.memmap(prob_path, dtype=np.float32, mode="r", shape=prob_shape)

    pit_values = []
    total_counts = np.zeros(len(EXPECTED_PROBS), dtype=np.int64)
    global_sse = 0.0
    land_sse = 0.0
    global_n = 0
    land_n = 0
    sum_error = 0.0
    sum_land_error = 0.0

    for i in tqdm(range(0, prob_shape[0], BATCH_SIZE), desc=f"eval T={temperature:.2f}, beta={beta:.3f}"):
        probs = np.asarray(prob_mm[i : i + BATCH_SIZE])
        y_batch = y_true_norm[i : i + BATCH_SIZE]
        probs_cal = calibrate_probs_map(probs, temperature, beta, bin_centers)

        pit_batch = probs_to_pit_values(probs_cal, y_batch, bin_centers, landmask)
        pit_values.append(pit_batch)
        total_counts += interval_counts_from_pit(pit_batch)

        mu_norm = (probs_cal * bin_centers[None, :, None, None]).sum(axis=1)
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

    pit_values = np.concatenate(pit_values)
    mace, observed = mace_from_pit(pit_values)

    return {
        "temperature": float(temperature),
        "beta": float(beta),
        "pit": pit_values,
        "observed": observed,
        "mace": mace,
        "mean_pit": float(pit_values.mean()),
        "global_rmse": float(np.sqrt(global_sse / max(global_n, 1))),
        "land_rmse": float(np.sqrt(land_sse / max(land_n, 1))),
        "mean_error": float(sum_error / max(global_n, 1)),
        "land_mean_error": float(sum_land_error / max(land_n, 1)),
    }


# ==============================================================================
# 7. PLOTTING
# ==============================================================================
def plot_fit_diagnostics(temp_only_hist, shift_hist):
    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    temps = [row["temperature"] for row in temp_only_hist]
    maces = [row["mace"] for row in temp_only_hist]
    ax[0].plot(temps, maces, marker="o", color="tab:blue")
    ax[0].set_title("Temperature-Only Fit", fontweight="bold")
    ax[0].set_xlabel("Temperature")
    ax[0].set_ylabel("Validation MACE")
    ax[0].grid(True, linestyle=":", alpha=0.5)

    betas = [row["beta"] for row in shift_hist["beta_center"]]
    pit_bias = [row["pit_bias"] for row in shift_hist["beta_center"]]
    ax[1].plot(betas, pit_bias, marker="o", color="tab:purple")
    ax[1].set_title("Beta Search at T=1", fontweight="bold")
    ax[1].set_xlabel("Beta")
    ax[1].set_ylabel("|mean PIT - 0.5|")
    ax[1].grid(True, linestyle=":", alpha=0.5)

    temps = [row["temperature"] for row in shift_hist["temp_refine"]]
    maces = [row["mace"] for row in shift_hist["temp_refine"]]
    ax[2].plot(temps, maces, marker="o", color="tab:green")
    ax[2].set_title("Final T Search with Beta Fixed", fontweight="bold")
    ax[2].set_xlabel("Temperature")
    ax[2].set_ylabel("Validation MACE")
    ax[2].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_reliability_triplet(raw, temp_only, shift_temp):
    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    variants = [
        ("Raw", raw, "steelblue"),
        ("Temp Only", temp_only, "seagreen"),
        ("Shift + Temp", shift_temp, "darkorange"),
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
            f"{title}\nT={result['temperature']:.3f}, beta={result['beta']:.3f}",
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

    temp_only_best, temp_only_hist = fit_temperature_only(point_probs, y_points, data["bin_centers"])
    shift_temp_best, shift_hist = fit_shift_temperature(point_probs, y_points, data["bin_centers"], temp_only_best)

    print("\nValidation-fit summary:")
    print(
        f"  Temp only:    T={temp_only_best['temperature']:.4f}  "
        f"MACE={temp_only_best['mace']:.4f}  mean PIT={temp_only_best['mean_pit']:.4f}"
    )
    print(
        f"  Shift+Temp:   T={shift_temp_best['temperature']:.4f}  "
        f"beta={shift_temp_best['beta']:.4f}  "
        f"MACE={shift_temp_best['mace']:.4f}  mean PIT={shift_temp_best['mean_pit']:.4f}"
    )

    plot_fit_diagnostics(temp_only_hist, shift_hist)

    test_prob_path, test_prob_shape = make_prob_memmap("test", data["X_test"], member_files)

    raw_result = evaluate_variant(
        test_prob_path, test_prob_shape, data["y_test"], data["bin_centers"],
        data["landmask"], data["y_mean"], data["y_std"], 1.0, 0.0
    )
    temp_result = evaluate_variant(
        test_prob_path, test_prob_shape, data["y_test"], data["bin_centers"],
        data["landmask"], data["y_mean"], data["y_std"], temp_only_best["temperature"], 0.0
    )
    shift_result = evaluate_variant(
        test_prob_path, test_prob_shape, data["y_test"], data["bin_centers"],
        data["landmask"], data["y_mean"], data["y_std"],
        shift_temp_best["temperature"], shift_temp_best["beta"]
    )

    print("\nTest split summary:")
    for label, result in [
        ("Raw", raw_result),
        ("Temp only", temp_result),
        ("Shift + Temp", shift_result),
    ]:
        print(
            f"  {label:12s} | MACE={result['mace']:.4f}  "
            f"mean PIT={result['mean_pit']:.4f}  "
            f"global RMSE={result['global_rmse']:.4f}  "
            f"land RMSE={result['land_rmse']:.4f}  "
            f"mean error={result['mean_error']:.4f}"
        )

    plot_reliability_triplet(raw_result, temp_result, shift_result)

    with open(calibration_out_path, "w") as f:
        json.dump(
            {
                "fit_land_points": int(len(y_points)),
                "temperature_grid": TEMP_GRID.tolist(),
                "beta_grid": BETA_GRID.tolist(),
                "temp_only_best": temp_only_best,
                "shift_temp_best": shift_temp_best,
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
                "test_shift_temp": {
                    "temperature": shift_result["temperature"],
                    "beta": shift_result["beta"],
                    "mace": shift_result["mace"],
                    "mean_pit": shift_result["mean_pit"],
                    "global_rmse": shift_result["global_rmse"],
                    "land_rmse": shift_result["land_rmse"],
                    "mean_error": shift_result["mean_error"],
                    "land_mean_error": shift_result["land_mean_error"],
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
