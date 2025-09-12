# scripts/ingest.py

import os
import uuid
import hashlib
import json
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from sqlalchemy import create_engine, text
from loguru import logger

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

INGEST_VERSION = "v1"

# -----------------------------
# Engine and filesystem
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
# Helper functions
# -----------------------------
def checksum_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_profile(ds, profile_idx=0):
    """Extracts metadata and measurements for one profile from the dataset."""
    profile = {}

    # ---------------------------
    # Helper function for safe extraction
    # ---------------------------
    def get_var(var_name, default=None):
        """Extract a variable value from ds, handling n_prof dims."""
        if var_name in ds:
            val = ds[var_name].values
            if 'n_prof' in ds[var_name].dims:
                return val[profile_idx]
            return val
        return default

    # ---------------------------
    # Float ID
    # ---------------------------
    fid = get_var('platform_number', '')
    if isinstance(fid, (bytes, np.bytes_)):
        fid = fid.decode('utf-8').strip()
    elif isinstance(fid, np.ndarray) and fid.size > 0:
        # handle array of bytes/strings
        val = fid.item() if fid.shape == () else fid[profile_idx]
        if isinstance(val, (bytes, np.bytes_)):
            fid = val.decode('utf-8').strip()
        else:
            fid = str(val).strip()
    profile['float_id'] = str(fid).strip()

    # ---------------------------
    # Profile coordinates and cycle
    # ---------------------------
    profile['lat'] = float(get_var('latitude', np.nan))
    profile['lon'] = float(get_var('longitude', np.nan))
    cyc = get_var('cycle_number', -1)
    try:
        profile['cycle_number'] = int(cyc)
    except Exception:
        profile['cycle_number'] = -1

    # ---------------------------
    # Profile date (from juld)
    # ---------------------------
    juld_val = get_var('juld')
    profile['profile_date'] = None
    if juld_val is not None:
        try:
            if isinstance(juld_val, np.datetime64):
                profile['profile_date'] = pd.to_datetime(juld_val).to_pydatetime()
            elif isinstance(juld_val, (int, float, np.number)):
                # If it's numeric, interpret relative to 1950-01-01
                profile['profile_date'] = datetime(1950, 1, 1) + pd.to_timedelta(juld_val, unit='D')
            else:
                profile['profile_date'] = pd.to_datetime(juld_val).to_pydatetime()
        except Exception:
            profile['profile_date'] = None

    # ---------------------------
    # Measurements & QC
    # ---------------------------
    measurements = {}
    qc_summary = {}
    var_map = {
        'PRES': ['pres'],
        'TEMP': ['temp'],
        'PSAL': ['psal'],
    }

    for std_var, candidates in var_map.items():
        for name in candidates:
            if name in ds.data_vars:
                vals = ds[name].values
                if 'n_prof' in ds[name].dims:
                    vals = vals[profile_idx]
                vals = np.asarray(vals).squeeze()
                fill = ds[name].attrs.get('_FillValue', np.nan)
                vals = np.where(vals == fill, np.nan, vals)
                measurements[std_var] = np.where(np.isfinite(vals), vals, np.nan).tolist()

                # QC variable (lowercase + "_qc")
                qc_name = f"{name}_qc"
                if qc_name in ds.data_vars:
                    qc_vals = ds[qc_name].values
                    if 'n_prof' in ds[qc_name].dims:
                        qc_vals = qc_vals[profile_idx]
                    qc_vals = np.asarray(qc_vals).squeeze()

                    # Handle byte and str flags
                    good_flags = np.sum((qc_vals == b'1') | (qc_vals == '1'))
                    qc_summary[std_var] = round((good_flags / len(qc_vals)) * 100) if len(qc_vals) > 0 else 0
                break  # stop after first matching candidate

    profile.update(measurements)
    profile['qc_summary'] = qc_summary
    return profile


def compute_summary(profile):
    """Returns a text summary of the profile."""
    lat, lon = profile.get('lat',0), profile.get('lon',0)
    date_str = profile['profile_date'].strftime("%Y-%m-%d") if profile.get('profile_date') else 'N/A'
    summary = f"Float {profile.get('float_id')} cycle {profile.get('cycle_number')} on {date_str} at ({lat:.2f},{lon:.2f})."

    for var in ['TEMP','PSAL','DOXY']:
        arr = np.array(profile.get(var,[]),dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size>0:
            qc = profile['qc_summary'].get(var,0)
            summary += f" {var}: mean {finite.mean():.2f}, range {finite.min():.2f}-{finite.max():.2f} ({qc}% QC)."
    return summary


def write_parquet(profile, profile_id, parquet_uri):
    df = pd.DataFrame({
        'profile_id':[str(profile_id)],
        'float_id':[profile.get('float_id')],
        'cycle_number':[profile.get('cycle_number')],
        'profile_date':[profile.get('profile_date')],
        'lat':[profile.get('lat')],
        'lon':[profile.get('lon')],
        **{k:[v] for k,v in profile.items() if isinstance(v,list)}
    })
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_uri, filesystem=s3_fs)
    logger.info(f"Parquet written to {parquet_uri}")


def upsert_db(profile, profile_id, parquet_uri, checksum, source_file):
    """Upserts float and profile metadata to Postgres"""
    lat, lon = profile.get('lat'), profile.get('lon')

    # Skip invalid coordinates
    if lat is None or lon is None or not np.isfinite(lat) or not np.isfinite(lon):
        logger.warning(f"Skipping profile {profile_id} due to invalid coordinates: lat={lat}, lon={lon}")
        return

    geom = f"POINT({lon} {lat})"
    pres_arr = np.array(profile.get('PRES',[]),dtype=float)
    finite_pres = pres_arr[np.isfinite(pres_arr)]

    with engine.begin() as conn:
        if profile.get('float_id'):
            conn.execute(text("""
                INSERT INTO floats(float_id) VALUES(:fid)
                ON CONFLICT (float_id) DO NOTHING
            """), {'fid': profile['float_id']})

        conn.execute(text("""
            INSERT INTO profiles(profile_id,float_id,cycle_number,profile_date,lat,lon,geom,
                                 n_levels,min_pres,max_pres,qc_summary,summary_text,parquet_path,
                                 source_file,source_checksum,ingest_version)
            VALUES(:pid,:fid,:cyc,:date,:lat,:lon,ST_GeomFromText(:geom,4326),
                   :n_levels,:min_pres,:max_pres,CAST(:qc AS jsonb),:summary,:parquet,:src,:cs,:ver)
            ON CONFLICT (profile_id) DO UPDATE
            SET lat=EXCLUDED.lat, lon=EXCLUDED.lon, geom=EXCLUDED.geom,
                n_levels=EXCLUDED.n_levels, min_pres=EXCLUDED.min_pres, max_pres=EXCLUDED.max_pres,
                qc_summary=EXCLUDED.qc_summary, summary_text=EXCLUDED.summary_text,
                parquet_path=EXCLUDED.parquet_path, source_file=EXCLUDED.source_file,
                source_checksum=EXCLUDED.source_checksum, ingest_version=EXCLUDED.ingest_version,
                updated_at=now()
        """), {
            'pid': str(profile_id),
            'fid': profile['float_id'],
            'cyc': profile.get('cycle_number'),
            'date': profile.get('profile_date'),
            'lat': lat,
            'lon': lon,
            'geom': geom,
            'n_levels': len(finite_pres),
            'min_pres': float(np.min(finite_pres)) if finite_pres.size>0 else None,
            'max_pres': float(np.max(finite_pres)) if finite_pres.size>0 else None,
            'qc': json.dumps(profile['qc_summary']),
            'summary': profile['summary_text'],
            'parquet': parquet_uri,
            'src': source_file,
            'cs': checksum,
            'ver': INGEST_VERSION
        })


def ingest_one_file(nc_file):
    checksum = checksum_file(nc_file)
    source_file = os.path.basename(nc_file)
    log_id = None

    try:
        # Check for idempotency
        with engine.begin() as conn:
            result = conn.execute(text(
                "SELECT id,status FROM process_log WHERE file_name=:f AND checksum=:c"
            ), {'f':source_file,'c':checksum}).first()
            if result and result.status=='OK':
                logger.info(f"File {source_file} already processed. Skipping.")
                return

            # Insert log
            log_id = conn.execute(text("""
                INSERT INTO process_log(file_name, checksum, status, started_at)
                VALUES(:f,:c,'PROCESSING',now()) RETURNING id
            """), {'f':source_file,'c':checksum}).scalar_one()

        logger.info(f"Ingesting {nc_file} (log id {log_id})")
        ds = xr.open_dataset(nc_file, mask_and_scale=True)
        n_prof = ds.dims.get('N_PROF',1)

        for i in range(n_prof):
            profile = extract_profile(ds,i)
            profile['summary_text'] = compute_summary(profile)
            profile_id = uuid.uuid4()
            year = profile['profile_date'].year if profile.get('profile_date') else 'unknown'
            month = f"{profile['profile_date'].month:02d}" if profile.get('profile_date') else '00'
            parquet_file = f"{profile.get('float_id','unknown')}_cycle_{profile.get('cycle_number','unknown')}_{profile_id}.parquet"
            parquet_uri = f"{PARQUET_BASE_URI}/year={year}/month={month}/{parquet_file}"

            write_parquet(profile, profile_id, parquet_uri)
            upsert_db(profile, profile_id, parquet_uri, checksum, source_file)

        # Update log as OK
        with engine.begin() as conn:
            conn.execute(text("UPDATE process_log SET status='OK', records_parsed=:n, ended_at=now() WHERE id=:id"),
                         {'n': n_prof, 'id': log_id})

        logger.success(f"Successfully ingested {n_prof} profiles from {nc_file}")

    except Exception as e:
        logger.error(f"Error ingesting {nc_file}: {e}")
        with engine.begin() as conn:
            if log_id:
                conn.execute(text("UPDATE process_log SET status='ERROR', error_msg=:err, ended_at=now() WHERE id=:id"),
                             {'err': str(e), 'id': log_id})
            else:
                conn.execute(text("INSERT INTO process_log(file_name, checksum, status, error_msg, ended_at) VALUES(:f,:c,'ERROR',:err,now())"),
                             {'f': source_file,'c': checksum,'err': str(e)})


# Main CLI
# -----------------------------
if __name__ == "__main__":
    import sys
    import glob

    # ingest all the nc files present in the samples folders

    if len(sys.argv) < 2:
        logger.error("Usage: python scripts/ingest.py <netcdf_file|folder>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        # Batch mode: process all .nc files in the folder
        nc_files = glob.glob(os.path.join(target, "*.nc"))
        logger.info(f"Found {len(nc_files)} NetCDF files in {target}")

        if not nc_files:
            logger.warning(f"No NetCDF files found in {target}")
            sys.exit(0)

        for nc_file in nc_files:
            ingest_one_file(nc_file)

        logger.success(f"Batch ingestion complete. {len(nc_files)} files processed.")

    else:
        # Single file mode
        ingest_one_file(target)
