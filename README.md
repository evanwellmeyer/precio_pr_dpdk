# precip_pr_dpdk

Probabilistic U-Net for gridded precipitation-change prediction. Given historical
precipitation climatology (`PR`), the model predicts a full per-pixel distribution
over `dPdK` (precipitation change per degree global warming, `mm/yr/K`) rather
than a single deterministic field.

---

## Overview

**Task:** Predict gridded `dPdK` from historical precipitation climatology:

- input: `PR` historical climatology (`mm/yr`)
- target: `dPdK` future precipitation change per degree warming (`mm/yr/K`)

**Method:** Softmax probabilistic regression. The network discretizes target space
into fixed bins, predicts a categorical distribution over those bins at each grid
cell, and trains against soft Gaussian-style labels centered on the normalized
true target.

**Network:** `ProbUNet` built on `Unet6R`:

- 6-level residual U-Net
- custom geo-aware padding
  - reflect padding in latitude
  - circular padding in longitude
- GroupNorm + Mish residual blocks
- `1x1` softmax head over `num_bins` output bins

**Output:** A tensor of per-bin probabilities with shape `(B, num_bins, H, W)`.
For evaluation, the expected value of that distribution is mapped back to physical
`dPdK` units.

---

## Files

### Core

| File | Purpose |
|------|---------|
| `unet.py` | Shared model definitions: `CustomPad`, `ConvResBlockSingle`, `Unet6R`, `SoftmaxHead`, `ProbUNet` |
| `train_pr_dpdk.py` | Main ensemble training script for HadGEM `PR -> dPdK` |
| `post_pr_dpdk.py` | Ensemble evaluation, baseline comparison, and RMSE summary export |
| `train_pr_dpdk_cv.py` | Leave-one-PPE-out cross-validation training (3 folds: GA7, GA8, GA9) |
| `post_pr_dpdk_cv.py` | Cross-validation post-analysis; evaluates each fold's held-out test PPE |

### Data Processing

| File | Purpose |
|------|---------|
| `make_cesm2_dpdK.py` | Build `CESM2_dPdK_rg128.nc` from CESM2 PPE SST4K precipitation and historical/dT fields |

### Diagnostics / Exploration

| File | Purpose |
|------|---------|
| `maps.ipynb` | Spatial plotting and map diagnostics |
| `calibration.ipynb` | Reliability / calibration checks for the probabilistic output |
| `PDFs.ipynb` | Distribution and probability-density diagnostics |
| `gaussian_weighting.ipynb` | Soft-label / Gaussian weighting experiments |
| `gridpoint_regression.ipynb` | Pointwise regression exploration |
| `correlations.ipynb` | Correlation-based diagnostics and comparisons |

---

## Pipeline

```text
RAW DATA
├── HadGEM/GA789_PR_his_rg128.nc
├── HadGEM/GA789_dPdK_rg128.nc
├── HadGEM/hadgem_landmask_rg128.nc
└── CESM2/SST4K/... PR files + CESM2 historical/dT fields

    ↓  make_cesm2_dpdK.py  (optional external utility)

DERIVED TARGETS
└── CESM2/CESM2_dPdK_rg128.nc

    ↓  train_pr_dpdk.py                  ↓  train_pr_dpdk_cv.py (optional)

TRAINING ARTIFACTS                        CV TRAINING ARTIFACTS
└── weights/unet_ens_.../                 └── weights/unet_cv_.../fold_{GA7,GA8,GA9}/
    ├── data_splits.npz                       ├── data_splits.npz
    ├── norm_stats.json                       ├── norm_stats.json
    ├── born_bins.json                        ├── born_bins.json
    ├── best_member{i}.pth                    ├── best_member{i}.pth
    └── <ens_name>_member{i}.pth              └── <cv_name>_member{i}.pth

    ↓  post_pr_dpdk.py                        ↓  post_pr_dpdk_cv.py

ANALYSIS OUTPUTS
└── weights/unet_ens_.../
    ├── softmax_ensemble_analysis_results.json
    └── softmax_ensemble_analysis_arrays.npz

    ↓  notebooks

MAPS / CALIBRATION / PDFS / FIGURES
```

---

## Data Assumptions

The scripts currently assume local data live under:

```text
/Users/ewellmeyer/Documents/research/HadGEM
/Users/ewellmeyer/Documents/research/CESM2
/Users/ewellmeyer/Documents/research/weights
```

### HadGEM Training Data

`train_pr_dpdk.py` expects:

- `GA789_PR_his_rg128.nc`
- `GA789_dPdK_rg128.nc`

These are read as:

- `PR`: historical precipitation climatology
- `dPdK`: target precipitation change per degree warming

### CESM2 Utility Input

`make_cesm2_dpdK.py` expects:

- future SST4K precipitation files under `CESM2/SST4K/PR/`
- `CESM2_PR_his_rg128.nc`
- `CESM2_dT_rg128.nc`

It produces:

- `CESM2_dPdK_rg128.nc`

That script is not part of the main HadGEM training pipeline, but it provides a
clean way to construct an out-of-sample `dPdK` dataset in the same grid/units.

---

## Model Architecture

**ProbUNet** (`unet.py`)

- Input: historical precipitation field `PR` → `(B, 1, H, W)`
- Backbone: `Unet6R`
- Head: per-pixel categorical distribution over `num_bins`
- Output: probability tensor `(B, num_bins, H, W)`

### Unet6R

- 6 encoder levels
- constant channel width `base_channels` throughout encoder, bottleneck, and decoder
- decoder mirrors the encoder with skip connections
- residual blocks use:
  - Conv
  - GroupNorm
  - Mish
  - optional Dropout

### Geo-Aware Padding

Global fields need different behavior in latitude and longitude:

- latitude: reflect padding
- longitude: circular padding

This avoids artificial edge effects at the dateline while preserving sensible
boundary behavior near the poles.

### Internal Padding

The U-Net pads inputs internally so spatial dimensions are divisible by `2^6 = 64`,
then crops the output back to the original grid. This lets the model run cleanly
on the `rg128` grid without manual preprocessing.

---

## Probabilistic Formulation

The target is not trained as a single scalar regression output. Instead:

1. Define a fixed set of physical target bins over `dPdK`
2. Normalize those bin centers using HadGEM train-only target statistics
3. Build soft labels around the normalized true target
4. Train the model to predict a probability distribution over bins

### Soft Labels

For each grid cell, the true normalized target is converted into a soft target:

```text
y_soft(b) ∝ exp[-0.5 * ((y_true - bin_center_b) / sigma_b)^2]
```

where:

- `bin_center_b` is the normalized bin center
- `sigma_b` comes from local bin spacing times `sigma_scale`

The final soft label is normalized to sum to 1 across bins.

### Loss

The training objective is the soft categorical negative log-likelihood:

```text
L = - mean_{pixels,bins} [ y_soft * log(p_pred) ]
```

In the current script:

- probabilities are clamped before `log`
- AMP is disabled by default on MPS for numerical stability

### Prediction

At evaluation time, the expected target is reconstructed as:

```text
E[dPdK] = sum_b p(b) * bin_center_b
```

then mapped from normalized units back to physical `mm/yr/K`.

---

## Training

### Main Script

Run from this repo directory:

```bash
python train_pr_dpdk.py
```

The training script does the following:

1. loads HadGEM `PR` and `dPdK`
2. creates a random train / val / test split
3. saves the split indices to `data_splits.npz`
4. computes normalization statistics from the train split only
5. saves normalization metadata to `norm_stats.json`
6. defines fixed target bins and soft-label widths
7. saves bin metadata to `born_bins.json`
8. trains an ensemble of probabilistic U-Nets with early stopping
9. exports checkpoints and final member weights

### Split Strategy

The current split is sample-wise and deterministic with `random_state=42`:

- train: `70%`
- validation: `10%`
- test: `20%`

Implementation detail:

- first split off `30%` from train
- then split that temporary set into val/test at `1/3` vs `2/3`

### Current Default Hyperparameters

From `train_pr_dpdk.py`:

```text
ensemble_size = 10
base_ch       = 200
gn_groups     = 1
k_size        = 3
pdrop         = 0.0
num_bins      = 64
sigma_scale   = 0.6
batch_train   = 10
batch_val     = 40
num_epochs    = 5000
patience      = 15
grad_clip     = 1.0
optimizer     = Adam(lr=1e-3, weight_decay=1e-5)
target range  = [-700, 1200] mm/yr/K
```

### Saved Outputs

Each experiment writes to a configuration-specific directory in:

```text
/Users/ewellmeyer/Documents/research/weights
```

The directory name encodes the active hyperparameters, for example:

```text
unet_ens_HG789_PR_dPdK_Softmax_unet6R_ch200_k3_128x_dPbins64_gn1_dpmin-700_dPmax1200
```

The training script saves:

- `data_splits.npz`
- `norm_stats.json`
- `born_bins.json`
- `best_member{i}.pth`
- `<experiment_name>_member{i}.pth`

`best_member{i}.pth` contains optimizer state and the best validation checkpoint.
The final exported member file is reloaded from the best checkpoint before saving.

---

## Post-Processing

### Main Script

```bash
python post_pr_dpdk.py
```

This script:

1. loads the saved ensemble members
2. reloads train/val/test splits and normalization metadata
3. reconstructs expected-value `dPdK` predictions from predicted probabilities
4. computes RMSE for:
   - each member
   - the ensemble mean
   - a simple PPE baseline
5. computes both:
   - global RMSE
   - land-only RMSE
6. saves compact outputs for later plotting

### PPE Baseline

The baseline is the train-mean target field:

```text
ppe_mean_dP = mean_train(dPdK)
```

This gives a simple “predict the mean field everywhere” benchmark against which
the neural ensemble is compared.

### Outputs

`post_pr_dpdk.py` writes:

- `softmax_ensemble_analysis_results.json`
- `softmax_ensemble_analysis_arrays.npz`

These include:

- file/sample ids
- train/val/test indices
- PPE RMSE
- ensemble RMSE
- member RMSE
- land-only variants of the same metrics
- saved lat weights and land mask in the NPZ bundle

---

## CESM2 dPdK Utility

`make_cesm2_dpdK.py` computes `dPdK` for CESM2 PPE members from SST4K forcing experiments.

### Formula

For each member:

```text
dPR      = future_PR - historical_PR
global_dT = area-weighted global-mean warming
dPdK     = dPR / global_dT
```

### Processing Steps

The script:

1. loads future precipitation in native CESM2 units
2. time-means each member
3. converts units to `mm/yr`
4. regrids to the HadGEM `rg128` grid with `xesmf`
5. computes per-member `dPdK`
6. saves `CESM2_dPdK_rg128.nc`

This is useful for external testing or comparing the HadGEM-trained model against
a second ensemble in a consistent target space.

---

## Notebooks

The notebooks are the diagnostics layer of the project rather than core pipeline code.
They are intended for:

- spatial map generation
- distribution / PDF checks
- calibration analysis
- pointwise regression experiments
- correlation diagnostics

The exact notebook contents are exploratory and may evolve faster than the core scripts.

---

## Running The Repo

Typical workflow:

```bash
python train_pr_dpdk.py
python post_pr_dpdk.py
```

Cross-validation workflow (leave-one-PPE-out):

```bash
python train_pr_dpdk_cv.py
python post_pr_dpdk_cv.py
```

Optional CESM2 preprocessing:

```bash
python make_cesm2_dpdK.py
```

Then open the notebooks for figures and diagnostics.

---

## Notes And Caveats

- Paths are currently hard-coded to the local workstation layout.
- The code is written as research scripts, not a packaged CLI tool.
- Training and post-processing configs must match. If you change:
  - `base_channels`
  - `gn_groups`
  - `kernel_size`
  - `num_bins`
  - target bin range
  then update both scripts accordingly.
- The output metadata filename is currently `born_bins.json`; that name is kept as-is
  because the scripts already depend on it.
- Some variable names still use legacy `dP_*` naming even though the target is `dPdK`.
  The README uses `dPdK` terminology consistently, but the code still contains a few
  older names for bin-range variables.
- AMP is disabled in the training script by default on MPS because the softmax/log-loss
  path is more stable in full precision.

---

## Dependencies

The scripts require a Python environment with at least:

- `numpy`
- `xarray`
- `torch`
- `scikit-learn`
- `xesmf` for `make_cesm2_dpdK.py`

No pinned environment file is currently included in this repo.
