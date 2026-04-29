# precip_pr_dpdk

Probabilistic U-Net workflows for gridded precipitation-change prediction from
historical precipitation climatology. The core model predicts a full per-pixel
distribution over a discretized target axis instead of a single deterministic
field.

This repo currently contains three related tracks:

1. a standard HadGEM `PR -> dPdK` ensemble training script
2. a matched leave-one-PPE-out `PR -> dPdK` cross-validation pipeline
3. analysis notebooks centered on the current manuscript `dPdK` run with `ch=128`

---

## What Is In Here

### Shared model code

| File | Role |
|------|------|
| `unet.py` | Shared model definitions: `CustomPad`, `ConvResBlockSingle`, `Unet6R`, `SoftmaxHead`, `ProbUNet` |

### Standard ensemble workflow

| File | Role | Current default target |
|------|------|------------------------|
| `train_pr_dpdk.py` | Standard HadGEM ensemble training | `dPdK` |
| `post_pr_dpdk.py` | Standalone post-analysis for a saved ensemble | `dPdK` |

### Cross-validation workflow

| File | Role |
|------|------|
| `train_pr_dpdk_cv.py` | Leave-one-PPE-out training over `GA7`, `GA8`, `GA9` |
| `post_pr_dpdk_cv.py` | Matched CV evaluation on the held-out PPE |

### Utilities

| File | Role |
|------|------|
| `make_cesm2_dpdK.py` | Build `CESM2_dPdK_rg128.nc` from CESM2 SST4K precipitation and historical/dT fields |

### Notebooks

| File | Role | Current state |
|------|------|---------------|
| `calibration.ipynb` | PIT / reliability / post-hoc calibration checks | Updated |
| `maps.ipynb` | Spatial diagnostics and uncertainty maps | Updated |
| `PDFs.ipynb` | Improvement-distribution figure | Simplified to the first figure only |
| `gaussian_weighting.ipynb` | Soft-label experiments | Exploratory |
| `gridpoint_regression.ipynb` | Pointwise baseline experiments | Exploratory |
| `correlations.ipynb` | Correlation diagnostics | Exploratory |
| `channel_sweep.ipynb` | Channel-count comparisons | Exploratory |

---

## Current Repo Status

### 1. Shared architecture

The model now lives in `unet.py` and is imported by the scripts and notebooks.
`ProbUNet` wraps a 6-level residual U-Net plus a `1x1` softmax head.

`Unet6R` supports two channel layouts:

- `pyramid=False`: flat-width U-Net, all levels use `base_channels`
- `pyramid=True`: channels double down the encoder and contract in the decoder

### 2. Standard training script

`train_pr_dpdk.py` is the main non-CV training entry point. It currently trains
HadGEM `PR -> dPdK` with:

- flat `Unet6R`
- `base_ch = 128`
- `num_bins = 64`
- target range `[-700, 1200] mm/yr/K`
- `RAdam(lr=1e-3)`
- AMP disabled on MPS (`use_amp = False`)

### 3. Standard post-analysis script

`post_pr_dpdk.py` is currently configured for the manuscript standard ensemble:

- target file: `GA789_dPdK_rg128.nc`
- run name contains `dPdK`
- `base_channels = 128`
- target range `[-700, 1200]`

### 4. Cross-validation pipeline

The matched `dPdK` train/post pair in the repo right now is:

- `train_pr_dpdk_cv.py`
- `post_pr_dpdk_cv.py`

This workflow uses:

- `arch = "pyramid"`
- `base_ch = 20`
- target range `[-700, 1200]`
- held-out PPE folds for `GA7`, `GA8`, `GA9`

### 5. Notebook analysis state

The updated notebooks are centered on the manuscript `dPdK` run with:

- `base_channels = 128`
- `num_bins = 64`
- `gn_groups = 1`
- target range `[-700, 1200]`

---

## Data Layout

The scripts assume a local workstation layout:

```text
/Users/ewellmeyer/Documents/research/HadGEM
/Users/ewellmeyer/Documents/research/CESM2
/Users/ewellmeyer/Documents/research/weights
```

### HadGEM files used in this repo

Depending on the script, the code expects some combination of:

- `GA789_PR_his_rg128.nc`
- `GA789_dPdK_rg128.nc`
- `GA789_dPdP_rg128.nc`
- `hadgem_landmask_rg128.nc`

### CESM2 files used by the utility

`make_cesm2_dpdK.py` expects:

- future SST4K precipitation files under `CESM2/SST4K/PR/`
- `CESM2_PR_his_rg128.nc`
- `CESM2_dT_rg128.nc`

and writes:

- `CESM2_dPdK_rg128.nc`

---

## Model And Target Formulation

### Inputs and outputs

- input: historical precipitation climatology `PR`
- target: either `dPdK` or `dPdP`, depending on the script
- model output: per-pixel categorical probabilities over fixed target bins

The output tensor has shape:

```text
(B, num_bins, H, W)
```

### Geo-aware padding

`CustomPad` uses:

- reflect padding in latitude
- circular padding in longitude

This is meant to reduce dateline artifacts while keeping sensible poleward
behavior.

### Internal padding and cropping

`Unet6R` pads fields internally so height and width are divisible by `2^6 = 64`,
then crops the output back to the original grid.

### Soft labels

Training is not scalar regression. The target is converted to a soft label over
the discretized target axis:

```text
y_soft(b) ∝ exp[-0.5 * ((y_true - c_b) / sigma_b)^2]
```

where:

- `c_b` is the normalized bin center
- `sigma_b` comes from local bin spacing times `sigma_scale`

These labels are symmetric around the true value on the normalized target axis.

### Loss

The scripts train with a soft categorical negative log-likelihood:

```text
L = - mean(y_soft * log p_pred)
```

`train_pr_dpdk.py` and `train_pr_dpdk_cv.py` clamp probabilities before taking
the log to avoid numerical issues.

### Prediction

The expected target field is reconstructed from the predicted probabilities:

```text
E[y] = sum_b p(b) c_b
```

then denormalized back to physical units.

---

## Standard HadGEM dPdK Training

Run:

```bash
python train_pr_dpdk.py
```

### What it does

1. loads HadGEM `PR` and `dPdK`
2. builds a random sample-wise train/val/test split with `random_state=42`
3. saves `data_splits.npz`
4. computes train-only normalization stats and saves `norm_stats.json`
5. defines fixed bins and soft-label widths and saves `born_bins.json`
6. trains an ensemble of `ProbUNet` members with early stopping
7. saves `best_member{i}.pth` plus exported member weights

### Current defaults from `train_pr_dpdk.py`

```text
ensemble_size = 10
base_seed     = 42
base_ch       = 128
gn_groups     = 1
k_size        = 3
pdrop         = 0.1
num_bins      = 64
sigma_scale   = 0.6
batch_train   = 40
batch_val     = 40
num_epochs    = 5000
patience      = 20
grad_clip     = 1.0
optimizer     = RAdam(lr=1e-3)
target range  = [-700, 1200]
AMP           = False
member loop   = range(0, ensemble_size)
```

### Output directory pattern

```text
unet_ens_HG789_PR_dPdK_Softmax_unet6R_ch{base_ch}_k{k_size}_128x_dPbins{num_bins}_gn{gn_groups}_dpmin{dP_min}_dPmax{dP_max}_sigma{sigma_scale}_dr{pdrop}
```

Example with current defaults:

```text
unet_ens_HG789_PR_dPdK_Softmax_unet6R_ch128_k3_128x_dPbins64_gn1_dpmin-700_dPmax1200_sigma0.6_dr0.1
```

Saved artifacts:

- `data_splits.npz`
- `norm_stats.json`
- `born_bins.json`
- `best_member{i}.pth`
- `<run_name>_member{i}.pth`

---

## Standalone Post-Analysis Scripts

### `post_pr_dpdk.py`

Run:

```bash
python post_pr_dpdk.py
```

What it does:

1. loads saved member checkpoints
2. reloads split and normalization metadata
3. reconstructs expected-value predictions
4. computes global and land-only RMSE
5. filters bad members using NaN/Inf checks plus an IQR outlier rule
6. writes compact JSON/NPZ outputs

Current hard-coded configuration:

```text
target       = dPdK
base_channels = 128
num_bins      = 64
target range  = [-700, 1200]
```

Outputs:

- `softmax_ensemble_analysis_results.json`
- `softmax_ensemble_analysis_arrays.npz`

These include:

- train/val/test indices
- member/global/land RMSE arrays
- retained member indices
- lat weights and land mask

The model is instantiated with dropout disabled for evaluation, but the run name
points to the `dr0.1` training ensemble.

### `post_pr_dpdk_cv.py`

Run:

```bash
python post_pr_dpdk_cv.py
```

This is the matched post-analysis for the CV training script. It:

- loads each held-out PPE fold
- computes PPE-baseline and ensemble RMSE
- filters bad members with the same NaN/Inf + IQR rule
- writes per-fold `cv_results.json` / `cv_results.npz`
- writes aggregate `cv_summary.json`

Current defaults:

```text
arch        = pyramid
base_ch     = 20
num_bins    = 64
target      = dPdK
target range = [-700, 1200]
```

---

## Cross-Validation Training

Run:

```bash
python train_pr_dpdk_cv.py
```

This workflow holds out one PPE family at a time:

- `GA7`
- `GA8`
- `GA9`

For each fold:

- the held-out PPE is test-only
- the other two PPEs are split into train/validation
- normalization and bins are recomputed from the fold training subset
- an ensemble is trained and saved under `fold_<PPE>`

### Current defaults from `train_pr_dpdk_cv.py`

```text
ensemble_size = 3
arch          = pyramid
base_ch       = 20
gn_groups     = 1
k_size        = 3
pdrop         = 0.0
num_bins      = 64
sigma_scale   = 0.6
batch_train   = 200
batch_val     = 100
num_epochs    = 5000
patience      = 20
grad_clip     = 1.0
val_fraction  = 0.2
optimizer     = RAdam(lr=1e-2, weight_decay=1e-5)
target range  = [-700, 1200]
```

Output directory pattern:

```text
unet_cv_HG789_PR_dPdK_Softmax_unet6R_{arch}_ch{base_ch}_k{k_size}_128x_dPbins{num_bins}_gn{gn_groups}_dpmin{dP_min}_dPmax{dP_max}/fold_{GA7|GA8|GA9}
```

---

## Notebook Status

### `calibration.ipynb`

This notebook is the current calibration workbench for the `dPdK`, `ch=128`
analysis run.

It now has three main stages:

1. raw PIT / reliability diagnostics
2. post-hoc recalibration experiments
3. global monotone CDF-warp experiments

Saved outputs:

- `temperature_calibration.json`
- `cdf_warp_calibration.json`

Current takeaway from the notebook workflow:

- symmetric residual width scaling is the useful post-hoc recalibration
- temperature-only scaling is worse
- global CDF warping improves PIT centering but hurts RMSE too much

### `maps.ipynb`

The map notebook has been updated to the same `dPdK`, `ch=128` analysis run.
It currently:

- loads `temperature_calibration.json`
- applies the validation-fit symmetric scale to uncertainty width
- optionally filters to `good_members` from `softmax_ensemble_analysis_results.json`
- prints both global and land-only summary metrics
- suppresses noisy `shapely` / `cartopy` plotting warnings

This notebook is the main place where the symmetric calibrated uncertainty width
is actually used downstream.

### `PDFs.ipynb`

This notebook has been simplified. It now focuses on the first figure only and
compares:

- Gaussian weighting
- gridpoint polynomial regression
- neural network (`ch=128`)

It is a distribution-figure notebook, not a calibration notebook. It is now
configured to use the shared 304-member NN test split for all three methods.

### Other notebooks

`gaussian_weighting.ipynb`, `gridpoint_regression.ipynb`, `correlations.ipynb`,
and `channel_sweep.ipynb` remain exploratory. They are useful for analysis, but
they are not the most actively maintained entry points in the repo.

---

## CESM2 Utility

Run:

```bash
python make_cesm2_dpdK.py
```

What it does:

1. loads CESM2 SST4K future precipitation files
2. time-means each file
3. converts native units to `mm/yr`
4. regrids to the HadGEM `rg128` grid with `xesmf`
5. computes `dPR = future - historical`
6. computes area-weighted global mean warming from `CESM2_dT_rg128.nc`
7. forms `dPdK = dPR / global_dT`
8. saves `CESM2_dPdK_rg128.nc`

This is separate from the HadGEM training loop, but useful for producing a
consistent out-of-sample `dPdK` dataset.

---

## Recommended Usage

### If you want to train a fresh standard HadGEM `dPdK` ensemble

```bash
python train_pr_dpdk.py
```

Then either:

- write a matching post script for that exact run, or
- edit `post_pr_dpdk.py` to point to the same target, architecture, and bin range

### If you want the matched train/post pipeline already present in the repo

```bash
python train_pr_dpdk_cv.py
python post_pr_dpdk_cv.py
```

### If you want the current figure-generation workflow

Use the notebooks, especially:

- `calibration.ipynb`
- `maps.ipynb`
- `PDFs.ipynb`

Those are currently aligned to the `dPdK`, `ch=128` analysis run and the
standard post-analysis script.

---

## Manuscript To Do

Current AIES manuscript state:

- The main Fig. 2 workflow has been rerun on the shared 304-member test split.
- Fig. 2 has been visually checked and looks good.
- Supplemental baseline figures Fig. S2, Fig. S3.1, and Fig. S3.2 have been
  updated.
- Gridpoint regression and Gaussian weighting now use the NN test members as
  held-out test data and fold the NN validation members into baseline training.
- `AMS LaTeX Package V6.1/figures/fig2.png` has been refreshed from the rerun
  baseline/PDF workflow.
- The Fig. 2 gridpoint baseline is the third-order polynomial regression. The
  manuscript now states this and notes that it is shown because it is the
  strongest tested local polynomial baseline.
- Fig. 3 seed diagnostics have been checked for the dropout 0.1 64-channel,
  128-channel, and 256-channel runs. The diagnostic figure was saved to
  `AMS LaTeX Package V6.1/figures/channel_sweep_seed_diagnostics.png`.
- The 128-channel run remains the practical choice: it has the best checked
  global, land-only, and in-range RMSE diagnostics. The 64-channel run is close
  but not better overall, and the 256-channel run does not show larger
  retained-seed spread or any meaningful secondary-metric gain. Keep Fig. 3
  simple.
- A cleaner seed-spread version of the channel sweep has been moved into the
  manuscript appendix as Fig. A1 (`AMS LaTeX Package V6.1/figures/figA1.png`),
  showing all three dropout settings and individual seed results. Fig. S4 is
  reserved for the leave-one-PPE-out cross-validation distributions.

Remaining items before submission:

- Do not rerun leave-one-PPE-out CV unless an advisor or reviewer specifically
  asks for a final-configuration generalization test. The current manuscript
  frames the existing CV as a diagnostic stress test, not as the source of the
  headline skill values.
- If CV does need to be rerun, run the final flat 128-channel, 10-seed setup
  first. Only run the pyramid architecture too if the paper is going to make an
  explicit architecture-robustness claim.
- Do one final pass on the promoted channel-sweep and archive-comparison
  material for narrative flow now that those diagnostics are in the main text.
- Add a GitHub release or archive DOI only if desired for final submission;
  the current text cites the public GitHub repository.

---

## Notes And Caveats

- Paths are hard-coded throughout the repo.
- This is a research-script repo, not a packaged CLI project.
- Training, post-processing, and notebooks are not all keyed off one central config.
- `gaussian_weighting.ipynb` and `gridpoint_regression.ipynb` are configured to
  use the NN 70/10/20 test split, with the NN validation members folded into
  baseline training.
- Some exploratory notebooks may still contain saved output from earlier runs and
  should be rerun before using their displayed values as final numbers.
- The metadata file is still named `born_bins.json`; that spelling is preserved
  because the scripts already depend on it.
- Several variables still use legacy `dP_*` naming even when the physical target
  is `dPdK`.

---

## Dependencies

The code has been run in a local `ml` conda environment on Python 3.10.

Core packages used across the repo include:

- `numpy`
- `xarray`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `torch`
- `netcdf4`
- `cartopy`
- `xesmf` for `make_cesm2_dpdK.py`

On Apple Silicon, the scripts are written to use `mps` when available.
`train_pr_dpdk.py` keeps AMP off by default for numerical stability.
