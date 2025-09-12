import os
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import s3fs
from dotenv import load_dotenv

# -----------------------------
# Load .env variables
# -----------------------------
load_dotenv()

# -----------------------------
# Connect to MinIO (S3)
# -----------------------------
s3_fs = s3fs.S3FileSystem(
    client_kwargs={
        "endpoint_url": f"http://{os.environ['MINIO_HOST']}:{os.environ['MINIO_PORT']}",
        "aws_access_key_id": os.environ["MINIO_ACCESS_KEY"],
        "aws_secret_access_key": os.environ["MINIO_SECRET_KEY"],
    }
)

# -----------------------------
# Read one Parquet back
# -----------------------------
parquet_path = (
    "argo-data/parquet/year=unknown/month=00/"
    "_cycle_-1_2096d746-953a-41ab-bf49-12b916e50303.parquet"
)

df = pd.read_parquet(f"s3://{parquet_path}", filesystem=s3_fs)
print("✅ Parquet read:")
print(df.head())

# -----------------------------
# Compare with original NetCDF
# -----------------------------
nc_file = "samples/nodc_D1901290_249.nc"
ds = xr.open_dataset(nc_file)

print("✅ NetCDF variables:", list(ds.data_vars.keys()))

# -----------------------------
# Extract clean values from NetCDF
# -----------------------------
def get_clean_values(varname):
    if varname not in ds:
        return None
    vals = ds[varname].values
    vals = np.asarray(vals).squeeze()
    fill = ds[varname].attrs.get("_FillValue", np.nan)
    vals = np.where(vals == fill, np.nan, vals)
    return vals

pres = get_clean_values("pres")
temp = get_clean_values("temp")
psal = get_clean_values("psal")

print("\n🔹 NetCDF summary:")
if temp is not None:
    print(f"TEMP: mean={np.nanmean(temp):.2f}, min={np.nanmin(temp):.2f}, max={np.nanmax(temp):.2f}")
if psal is not None:
    print(f"PSAL: mean={np.nanmean(psal):.2f}, min={np.nanmin(psal):.2f}, max={np.nanmax(psal):.2f}")
if pres is not None:
    print(f"PRES: min={np.nanmin(pres):.2f}, max={np.nanmax(pres):.2f}")

# -----------------------------
# Compare Parquet vs NetCDF (basic check)
# -----------------------------
print("\n🔹 Parquet summary:")
if "temp" in df:
    print(f"TEMP parquet: mean={df['temp'].mean():.2f}, min={df['temp'].min():.2f}, max={df['temp'].max():.2f}")
if "psal" in df:
    print(f"PSAL parquet: mean={df['psal'].mean():.2f}, min={df['psal'].min():.2f}, max={df['psal'].max():.2f}")
if "pres" in df:
    print(f"PRES parquet: min={df['pres'].min():.2f}, max={df['pres'].max():.2f}")

# -----------------------------
# Visualization: TEMP vs PRES
# -----------------------------
if pres is not None and temp is not None:
    plt.figure(figsize=(5, 7))
    plt.plot(temp, pres, "r.-", label="TEMP")
    plt.gca().invert_yaxis()  # depth increases downward
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Pressure (dbar)")
    plt.title("TEMP vs PRES (from NetCDF)")
    plt.legend()
    plt.tight_layout()
    plt.show()
