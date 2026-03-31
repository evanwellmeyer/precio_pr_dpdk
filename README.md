# precip_pr_dpdk

Probabilistic precipitation regression for HadGEM data using a softmax U-Net.

This repo trains an ensemble of convolutional models to predict a distribution over `dPdK` values from historical precipitation climatology (`PR`). The model outputs per-bin probabilities on a fixed target grid, and downstream analysis converts those probabilities into expected-value predictions and RMSE diagnostics.

## What This Repo Contains

- `train_pr_dpdk.py`
  Trains an ensemble of probabilistic U-Net models on HadGEM data. It:
  - loads historical precipitation (`PR`) and target change (`dPdK`)
  - computes train/val/test splits
  - normalizes inputs and targets using train-only statistics
  - builds soft labels over fixed `dP` bins
  - trains an ensemble with early stopping
  - saves weights and metadata needed for later analysis

- `post_pr_dpdk.py`
  Loads the saved ensemble and evaluates it on the full HadGEM sample set. It:
  - reconstructs expected-value predictions from bin probabilities
  - computes global and land-only RMSEs
  - compares the ensemble against a simple PPE mean baseline
  - writes compact JSON and NPZ summaries for plotting and analysis

- `unet.py`
  Shared model definitions:
  - `CustomPad`
  - `ConvResBlockSingle`
  - `Unet6R`
  - `SoftmaxHead`
  - `ProbUNet`

- notebooks
  A set of exploratory notebooks for maps, calibration, correlations, PDFs, and related diagnostics.

## Model Summary

The main model is `ProbUNet`, which combines:

- a 6-level residual U-Net backbone (`Unet6R`)
- custom geo-aware padding
  - reflect padding in latitude
  - circular padding in longitude
- a `1x1` convolutional softmax head over `num_bins` target bins

The network predicts a categorical distribution at each grid cell rather than a single scalar value. Training uses soft Gaussian-like targets centered on the normalized true `dPdK` value.

## Expected Data Layout

The scripts currently expect local files under:

```text
/Users/ewellmeyer/Documents/research/HadGEM
```

Training and analysis use:

```text
GA789_PR_his_rg128.nc
GA789_dPdK_rg128.nc
hadgem_landmask_rg128.nc
```

Weights and metadata are written under:

```text
/Users/ewellmeyer/Documents/research/weights
```

If you move the data or weights directory, update the hard-coded paths in:

- `train_pr_dpdk.py`
- `post_pr_dpdk.py`

## Python Requirements

The scripts rely on:

- `python`
- `numpy`
- `xarray`
- `torch`
- `scikit-learn`

This repo does not currently include a pinned environment file, so the active local environment needs those packages available.

## Training

Run from this repo directory:

```bash
python train_pr_dpdk.py
```

The training script will:

1. load HadGEM `PR` and `dPdK`
2. create and save train/val/test splits in `data_splits.npz`
3. save normalization statistics in `norm_stats.json`
4. save bin definitions in `born_bins.json`
5. train `ensemble_size` separate members
6. save:
   - best checkpoints as `best_member*.pth`
   - final exported members as `<model_name>_member*.pth`

### Current Training Configuration

The default configuration in `train_pr_dpdk.py` is:

- ensemble size: `10`
- base channels: `8`
- kernel size: `3`
- group norm groups: `1`
- number of bins: `64`
- bin range: `[-700, 1200]`
- batch size: `200` train / `100` val
- optimizer: `Adam`
- learning rate: `1e-3`
- AMP: disabled by default on MPS

AMP is currently turned off because the probabilistic softmax/log loss path is more stable in full precision on Apple Silicon.

## Post-Processing

After training, run:

```bash
python post_pr_dpdk.py
```

This script loads the ensemble and computes:

- member-wise RMSE
- ensemble-mean RMSE
- land-only RMSE
- PPE baseline RMSE

Outputs are saved into the same experiment directory as:

- `softmax_ensemble_analysis_results.json`
- `softmax_ensemble_analysis_arrays.npz`

## Saved Metadata

Each training run saves the files needed to reproduce analysis:

- `data_splits.npz`
  Saved train, validation, and test indices

- `norm_stats.json`
  Saved input and target normalization statistics

- `born_bins.json`
  Saved target-bin centers and per-bin soft-label widths

## Notes And Caveats

- Paths are hard-coded for the current workstation layout.
- The experiment directory name is built from the active training configuration. If you change model hyperparameters, bin ranges, or normalization assumptions, the output directory name will also change.
- `post_pr_dpdk.py` assumes its config matches the run you want to analyze. If you change the training config, update the same values in the post script before running it.
- The scripts are written as executable research scripts, not as a packaged library or CLI.

## Typical Workflow

```bash
python train_pr_dpdk.py
python post_pr_dpdk.py
```

Then use the notebooks in this repo for plotting, calibration checks, maps, and summary figures.
