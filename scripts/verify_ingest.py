import os
import pandas as pd
import s3fs
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
PG_HOST = os.environ.get("PG_HOST")
PG_PORT = os.environ.get("PG_PORT")
PG_USER = os.environ.get("PG_USER")
PG_PASSWORD = os.environ.get("PG_PASSWORD")
PG_DB = os.environ.get("PG_DB")
PG_CONN = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

MINIO_HOST = os.environ.get("MINIO_HOST")
MINIO_PORT = os.environ.get("MINIO_PORT")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY")
PARQUET_BASE_URI = os.environ.get("PARQUET_BASE_URI")

# -----------------------------
# Setup connections
# -----------------------------
engine = create_engine(PG_CONN, future=True)
s3_fs = s3fs.S3FileSystem(
    client_kwargs={
        "endpoint_url": f"http://{MINIO_HOST}:{MINIO_PORT}",
        "aws_access_key_id": MINIO_ACCESS_KEY,
        "aws_secret_access_key": MINIO_SECRET_KEY
    }
)

# -----------------------------
# Specify the file
# -----------------------------
nc_file = "samples/nodc_D1901290_249.nc"
parquet_prefix = PARQUET_BASE_URI + "/year=unknown/month=00/"

# -----------------------------
# 1️⃣ Verify process_log
# -----------------------------
with engine.begin() as conn:
    log = conn.execute(
        text("SELECT * FROM process_log WHERE file_name=:f ORDER BY started_at DESC LIMIT 1"),
        {"f": nc_file.split("/")[-1]}
    ).first()

if log:
    print("\n✅ process_log entry:")
    print(dict(log._mapping))
else:
    print("\n❌ No process_log entry found for this file.")

# -----------------------------
# 2️⃣ Verify Parquet files
# -----------------------------
print("\n🔹 Checking Parquet files in S3...")
files = s3_fs.ls(parquet_prefix)
parquet_files = [f for f in files if "_cycle_" in f]
if parquet_files:
    print(f"✅ Found {len(parquet_files)} Parquet file(s):")
    for f in parquet_files:
        print("  -", f)

    # Try reading the first one
    print("\n🔹 Reading first Parquet file:")
    df = pd.read_parquet(parquet_files[0], filesystem=s3_fs)
    print(df.head())
else:
    print("❌ No Parquet files found under", parquet_prefix)

# -----------------------------
# 3️⃣ Verify profiles table
# -----------------------------
with engine.begin() as conn:
    profiles = conn.execute(
        text("SELECT profile_id, float_id, lat, lon, summary_text FROM profiles ORDER BY created_at DESC LIMIT 5")
    ).all()

if profiles:
    print("\n✅ Last profiles inserted:")
    for p in profiles:
        print(dict(p._mapping))
else:
    print("\n❌ No profiles found in database.")



