from pathlib import Path

import numpy as np
import xarray as xr


DATA_FILE = Path("/Users/ewellmeyer/Documents/research/HadGEM/GA789_dPdK_rg128.nc")
VAR_NAME = "dPdK"
LOW = -700.0
HIGH = 1200.0


with xr.open_dataset(DATA_FILE) as ds:
    values = ds[VAR_NAME].values.astype(np.float64)

finite = np.isfinite(values)
data = values[finite]
covered = ((data >= LOW) & (data <= HIGH)).mean() * 100.0

print(f"File: {DATA_FILE}")
print(f"Variable: {VAR_NAME}")
print(f"Finite values: {data.size:,}")
print(f"Min: {data.min():.3f}")
print(f"Max: {data.max():.3f}")
print(f"Coverage in [{LOW:.0f}, {HIGH:.0f}]: {covered:.4f}%")
