"""
Compute CESM2 dP/dK fields (mm/yr per K warming) for each PPE member.

Steps:
  1. Load each future PR file (native 192x288, m/s), time-mean, regrid to rg128 (128x192)
  2. Load historic PR (already rg128, mm/yr) and dT (rg128, K spatial field)
  3. dPR = future_PR_mmyr - historic_PR_mmyr   (per member)
  4. global_dT = area-weighted mean of dT field (per member scalar)
  5. dPdK = dPR / global_dT
  6. Save CESM2_dPdK_rg128.nc
"""

import glob
import os
import re

import numpy as np
import xarray as xr
import xesmf as xe

CESM2_DIR = "/Users/ewellmeyer/Documents/research/CESM2"
SST4K_DIR = os.path.join(CESM2_DIR, "SST4K", "PR")
HIST_FILE = os.path.join(CESM2_DIR, "CESM2_PR_his_rg128.nc")
DT_FILE   = os.path.join(CESM2_DIR, "CESM2_dT_rg128.nc")
OUT_FILE  = os.path.join(CESM2_DIR, "CESM2_dPdK_rg128.nc")

S_PER_YR = 365.25 * 86400  # m/s -> mm/yr conversion factor (* 1000 already in mm)

future_files = sorted(glob.glob(os.path.join(SST4K_DIR, "cc_PPE_250_ensemble_SST4K.*.h0.PR.nc")))
print(f"Found {len(future_files)} future PR files")

ds_hist = xr.open_dataset(HIST_FILE)   # (realization, lat, lon), mm/yr
ds_dt   = xr.open_dataset(DT_FILE)     # (realization, lat, lon), K
print(f"Historic PR shape: {ds_hist['PR'].shape}")
print(f"dT shape:          {ds_dt['dT'].shape}")

realizations = ds_hist.realization.values
n_members = len(realizations)
lat_out = ds_hist.latitude.values
lon_out = ds_hist.longitude.values

# area weights on rg128 grid (cos-lat)
weights = np.cos(np.deg2rad(lat_out))
weights /= weights.mean()
# 2D weights for global mean: (lat, lon) -> normalise by total area
cos2d = np.cos(np.deg2rad(lat_out[:, None])) * np.ones((1, len(lon_out)))
cos2d /= cos2d.sum()

# extract member index from filename for matching to historic realization order
def member_idx(path):
    m = re.search(r'SST4K\.(\d+)\.h0', path)
    return int(m.group(1)) if m else -1

future_by_idx = {member_idx(f): f for f in future_files}

# build target grid dataset for xesmf
ds_out_grid = xr.Dataset({
    "lat": (["lat"], lat_out),
    "lon": (["lon"], lon_out),
})

dPdK_all = np.full((n_members, len(lat_out), len(lon_out)), np.nan, dtype=np.float32)

for i, real_id in enumerate(realizations):
    m_idx = int(real_id.split("_")[-1])   # e.g. CESM2_PPE_042 -> 42

    if m_idx not in future_by_idx:
        print(f"  WARNING: no future file for {real_id} (idx={m_idx}), skipping")
        continue

    ds_fut = xr.open_dataset(future_by_idx[m_idx])

    # time-mean and unit convert: m/s -> mm/yr
    pr_fut_mean = ds_fut["PREC"].mean("time") * S_PER_YR * 1000   # (lat, lon)

    # shift longitude 0-360 -> -180-180 to match rg128
    pr_fut_mean = pr_fut_mean.assign_coords(
        lon=(((pr_fut_mean.lon + 180) % 360) - 180)
    ).sortby("lon")

    # build source grid for xesmf (needs lat/lon named correctly)
    ds_in_grid = pr_fut_mean.to_dataset(name="PR").rename({"lat": "lat", "lon": "lon"})

    regridder = xe.Regridder(ds_in_grid, ds_out_grid, method="bilinear", reuse_weights=False)
    pr_fut_rg = regridder(ds_in_grid["PR"]).values   # (128, 192)

    pr_hist = ds_hist["PR"].sel(realization=real_id).values   # (128, 192), mm/yr

    dPR = pr_fut_rg - pr_hist

    # per-member global mean dT (area-weighted)
    dT_field = ds_dt["dT"].sel(realization=real_id).values   # (128, 192), K
    global_dT = float((dT_field * cos2d).sum())

    dPdK_all[i] = (dPR / global_dT).astype(np.float32)

    ds_fut.close()

    if (i + 1) % 25 == 0 or i == 0:
        print(f"  {i+1}/{n_members}  {real_id}  global_dT={global_dT:.3f} K")

# save
ds_out = xr.Dataset(
    {"dPdK": (["realization", "latitude", "longitude"], dPdK_all)},
    coords={
        "realization": realizations,
        "latitude":    lat_out,
        "longitude":   lon_out,
    },
)
ds_out["dPdK"].attrs = {
    "long_name": "Precipitation change per degree global warming (SST4K - historical)",
    "units": "mm/yr/K",
}
ds_out.to_netcdf(OUT_FILE)
print(f"\nSaved {OUT_FILE}")
print(f"dPdK shape: {ds_out['dPdK'].shape}")
print(f"dPdK mean (global): {float(np.nanmean(dPdK_all)):.4f} mm/yr/K")
