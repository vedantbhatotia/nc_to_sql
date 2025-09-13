import os
import uuid
import hashlib
import json
import shutil
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
import boto3
import botocore
from sqlalchemy import create_engine, text
from loguru import logger

load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
PG_HOST = os.environ.get("PG_HOST", "postgres")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_USER = os.environ.get("PG_USER", "argo")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "argo_pass")
PG_DB = os.environ.get("PG_DB", "argo_db")
PG_CONN = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

MINIO_HOST = os.environ.get("MINIO_HOST", "minio")
MINIO_PORT = os.environ.get("MINIO_PORT", "9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minio123")
PARQUET_BASE_URI = os.environ.get("PARQUET_BASE_URI", "s3://argo-data/parquet")

BUCKET_NAME = PARQUET_BASE_URI.replace("s3://", "").split("/")[0]
INGEST_VERSION = os.environ.get("INGEST_VERSION", "v1")
QUARANTINE_DIR = os.environ.get("QUARANTINE_DIR", "quarantine")
os.makedirs(QUARANTINE_DIR, exist_ok=True)

# -----------------------------
# Engine and S3 filesystem
# -----------------------------
engine = create_engine(PG_CONN, future=True)
s3_fs = s3fs.S3FileSystem(
    client_kwargs={
        "endpoint_url": f"http://{MINIO_HOST}:{MINIO_PORT}",
        "aws_access_key_id": MINIO_ACCESS_KEY,
        "aws_secret_access_key": MINIO_SECRET_KEY,
    }
)

# -----------------------------
# Helpers
# -----------------------------
def convert_numpy(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list)):
        return obj.tolist()
    return obj

def ensure_bucket_exists(bucket_name=BUCKET_NAME):
    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://{MINIO_HOST}:{MINIO_PORT}",
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )
    try:
        s3.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket {bucket_name} exists")
    except botocore.exceptions.ClientError:
        s3.create_bucket(Bucket=bucket_name)
        logger.success(f"Created bucket {bucket_name}")

def ensure_parquet_prefix(parquet_base_uri: str):
    if not parquet_base_uri.startswith("s3://"):
        raise AssertionError("PARQUET_BASE_URI must start with s3://")
    key = parquet_base_uri.replace("s3://", "").rstrip("/")
    try:
        s3_fs.makedirs(key, exist_ok=True)
    except Exception:
        try:
            s3_fs.mkdir(key)
        except Exception:
            logger.debug("Parquet prefix will be created on write.")

def checksum_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _choose_dim_name(ds, *candidates):
    for c in candidates:
        if c in ds.sizes: return c
        if c.upper() in ds.sizes: return c.upper()
        if c.lower() in ds.sizes: return c.lower()
    return None

def _get_var_case_insensitive(ds, *names):
    for n in names:
        if n in ds: return n, ds[n]
        if n.upper() in ds: return n.upper(), ds[n.upper()]
        if n.lower() in ds: return n.lower(), ds[n.lower()]
    return None, None

def canonicalize_float_id(raw_value, checksum: str) -> (str, str):
    fid = ""
    if isinstance(raw_value, (bytes, np.bytes_)):
        fid = raw_value.decode("utf-8", errors="ignore").strip()
    elif isinstance(raw_value, np.ndarray):
        try:
            fid_item = raw_value.item() if raw_value.shape == () else raw_value[0]
            if isinstance(fid_item, (bytes, np.bytes_)):
                fid = fid_item.decode("utf-8", errors="ignore").strip()
            else:
                fid = str(fid_item).strip()
        except Exception:
            fid = str(raw_value).strip()
    elif raw_value is not None:
        fid = str(raw_value).strip()
    if not fid:
        fid = f"unknown_{checksum[:8]}"
    return fid, str(raw_value) if raw_value is not None else None

# -----------------------------
# Profile extraction
# -----------------------------
def extract_profile(ds: xr.Dataset, profile_idx: int, checksum: str) -> dict:
    profile = {}
    def get_val(*var_names, default=None):
        name, da = _get_var_case_insensitive(ds, *var_names)
        if da is None: return default
        vals = da.values
        prof_dim = _choose_dim_name(ds, "N_PROF", "n_prof")
        if prof_dim and prof_dim in da.dims:
            try: return vals[profile_idx]
            except Exception: pass
        return vals
    fid_raw = get_val("platform_number", "PLATFORM_NUMBER")
    fid, fid_orig = canonicalize_float_id(fid_raw, checksum)
    profile["float_id"] = fid
    profile["float_id_raw"] = fid_orig
    lat = get_val("latitude", "LATITUDE", "lat")
    lon = get_val("longitude", "LONGITUDE", "lon")
    try: profile["lat"] = float(np.nan if lat is None else lat)
    except Exception: profile["lat"] = float(np.nan)
    try: profile["lon"] = float(np.nan if lon is None else lon)
    except Exception: profile["lon"] = float(np.nan)
    cyc = get_val("cycle_number", "CYCLE_NUMBER", "cycle_number")
    try: profile["cycle_number"] = int(cyc)
    except Exception: profile["cycle_number"] = -1
    juld = get_val("juld", "JULD", "juld_location")
    profile["profile_date"] = None
    if juld is not None:
        try:
            profile["profile_date"] = pd.to_datetime(juld).to_pydatetime()
        except Exception:
            try:
                profile["profile_date"] = datetime(1950,1,1) + pd.to_timedelta(float(juld), unit="D")
            except Exception:
                profile["profile_date"] = None
    def extract_array(var_candidates):
        name, da = _get_var_case_insensitive(ds, *var_candidates)
        if da is None: return []
        vals = da.values
        prof_dim = _choose_dim_name(ds, "N_PROF", "n_prof")
        if prof_dim and prof_dim in da.dims:
            try: vals = vals[profile_idx]
            except Exception: pass
        vals = np.asarray(vals).squeeze()
        fill = da.attrs.get("_FillValue", np.nan)
        vals = np.where(vals==fill, np.nan, vals)
        vals = vals.flatten() if vals.ndim>1 else vals
        return np.where(np.isfinite(vals), vals, np.nan).tolist()
    profile["PRES"] = extract_array(["PRES", "pres", "DEPTH", "depth"])
    profile["TEMP"] = extract_array(["TEMP", "temp", "TEMPF"])
    profile["PSAL"] = extract_array(["PSAL", "psal", "SALT"])
    profile["DOXY"] = extract_array(["DOXY", "doxy"])
    profile["CHLA"] = extract_array(["CHLA", "chla"])
    qc_summary = {}
    qc_vars = {
        "PRES": ["pres_qc","PRES_QC"],
        "TEMP": ["temp_qc","TEMP_QC"],
        "PSAL": ["psal_qc","PSAL_QC"],
        "DOXY": ["doxy_qc","DOXY_QC"],
        "CHLA": ["chla_qc","CHLA_QC"]
    }
    for std, candidates in qc_vars.items():
        name, da = _get_var_case_insensitive(ds, *candidates)
        if da is None:
            qc_summary[std]=None
            continue
        vals = da.values
        prof_dim = _choose_dim_name(ds, "N_PROF","n_prof")
        if prof_dim and prof_dim in da.dims:
            try: vals = vals[profile_idx]
            except Exception: pass
        vals = np.asarray(vals).squeeze()
        if vals.size==0: qc_summary[std]=None; continue
        good_flags = np.sum((vals==b"1")|(vals=="1")|(vals==1))
        qc_summary[std]=round((good_flags/vals.size)*100) if vals.size>0 else None
    profile["qc_summary"]=qc_summary
    return profile

def compute_summary(profile: dict) -> str:
    lat, lon = profile.get("lat", np.nan), profile.get("lon", np.nan)
    date_str = profile.get("profile_date").strftime("%Y-%m-%d") if profile.get("profile_date") else "N/A"
    summary = f"Float {profile.get('float_id','N/A')} cycle {profile.get('cycle_number','N/A')} on {date_str} at ({lat:.2f},{lon:.2f})."
    for var, unit in [("TEMP","°C"),("PSAL","PSU"),("DOXY","µmol/kg"),("CHLA","mg/m³")]:
        arr = np.array(profile.get(var, []), dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size>0:
            qc = profile.get("qc_summary", {}).get(var, 0) or 0
            summary += f" {var}: mean {np.nanmean(finite):.2f}{unit}, range {np.nanmin(finite):.2f}-{np.nanmax(finite):.2f}{unit} ({qc}% good)."
    return summary

# -----------------------------
# Ingest one file
# -----------------------------
def ingest_one_file(nc_file: str):
    checksum = checksum_file(nc_file)
    source_file = os.path.basename(nc_file)
    log_id = None
    ds = None
    try:
        with engine.begin() as conn:
            res = conn.execute(
                text("SELECT id,status FROM process_log WHERE file_name=:f AND checksum=:c"),
                {"f": source_file,"c":checksum}
            ).first()
            if res and res.status=="OK":
                logger.info(f"⏭ SKIP {source_file} already ingested")
                conn.execute(
                    text("INSERT INTO process_log(file_name, checksum, status, started_at, ended_at) VALUES(:f,:c,'SKIPPED',now(),now())"),
                    {"f": source_file,"c":checksum}
                )
                return
            log_id = conn.execute(
                text("INSERT INTO process_log(file_name, checksum, status, started_at) VALUES(:f,:c,'PROCESSING',now()) RETURNING id"),
                {"f": source_file,"c":checksum}
            ).scalar_one()

        logger.info(f"📥 Ingesting {nc_file}")
        ds = xr.open_dataset(nc_file, mask_and_scale=True)
        n_prof = int(ds.sizes.get("N_PROF") or ds.sizes.get("n_prof") or 1)

        float_meta = dict(ds.attrs) if ds.attrs else {}
        platform_type = float_meta.get("platform_type") or float_meta.get("PLATFORM_TYPE")
        wmo_id = float_meta.get("wmo_id") or float_meta.get("WMO_INST_TYPE")
        processed = 0

        for i in range(n_prof):
            profile = extract_profile(ds, i, checksum)
            profile["summary_text"] = compute_summary(profile)
            profile_id = uuid.uuid4()

            year = profile.get("profile_date").year if profile.get("profile_date") else "unknown"
            month = f"{profile['profile_date'].month:02d}" if profile.get("profile_date") else "00"
            parquet_file = f"{profile.get('float_id','unknown')}_cycle_{profile.get('cycle_number','unknown')}_{profile_id}.parquet"
            parquet_uri = f"{PARQUET_BASE_URI.rstrip('/')}/year={year}/month={month}/{parquet_file}"

            df = pd.DataFrame({
                "profile_id":[str(profile_id)],
                "float_id":[profile.get("float_id")],
                "cycle_number":[profile.get("cycle_number")],
                "profile_date":[profile.get("profile_date")],
                "lat":[profile.get("lat")],
                "lon":[profile.get("lon")],
                "summary_text":[profile.get("summary_text")],
                "qc_summary":[json.dumps(profile.get("qc_summary",{}), default=convert_numpy)],
                "pres":[profile.get("PRES",[])],
                "temp":[profile.get("TEMP",[])],
                "psal":[profile.get("PSAL",[])],
                "doxy":[profile.get("DOXY",[])],
                "chla":[profile.get("CHLA",[])]
            })

            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, parquet_uri, filesystem=s3_fs)
            logger.info(f"✅ Wrote Parquet: {parquet_uri}")

            lat, lon = profile.get("lat"), profile.get("lon")
            if lat is None or lon is None or not np.isfinite(lat) or not np.isfinite(lon):
                logger.warning(f"Skipping DB insert for profile {profile_id} due to invalid coords")
                processed += 1
                continue

            with engine.begin() as conn:
                if profile.get("float_id"):
                    conn.execute(text("""
                        INSERT INTO floats(float_id, platform_type, wmo_id, metadata, created_at, updated_at)
                        VALUES(:fid,:ptype,:wmo,CAST(:meta AS JSONB),now(),now())
                        ON CONFLICT (float_id) DO UPDATE
                        SET platform_type=COALESCE(EXCLUDED.platform_type,floats.platform_type),
                            wmo_id=COALESCE(EXCLUDED.wmo_id,floats.wmo_id),
                            metadata=EXCLUDED.metadata,
                            updated_at=now()
                    """), {"fid":profile.get("float_id"),"ptype":platform_type,"wmo":wmo_id,"meta":json.dumps(float_meta,default=convert_numpy)})

                conn.execute(text("""
                    INSERT INTO profiles(profile_id,float_id,cycle_number,profile_date,lat,lon,geom,
                                         n_levels,min_pres,max_pres,qc_summary,summary_text,parquet_path,
                                         source_file,source_checksum,ingest_version,created_at,updated_at)
                    VALUES(:pid,:fid,:cyc,:date,:lat,:lon,ST_GeomFromText(:geom,4326),
                           :n_levels,:min_pres,:max_pres,CAST(:qc AS JSONB),:summary,:parquet,
                           :src,:cs,:ver,now(),now())
                    ON CONFLICT (float_id, cycle_number) DO UPDATE
                      SET profile_date=EXCLUDED.profile_date,
                          lat=EXCLUDED.lat,
                          lon=EXCLUDED.lon,
                          geom=EXCLUDED.geom,
                          n_levels=EXCLUDED.n_levels,
                          min_pres=EXCLUDED.min_pres,
                          max_pres=EXCLUDED.max_pres,
                          qc_summary=EXCLUDED.qc_summary,
                          summary_text=EXCLUDED.summary_text,
                          parquet_path=EXCLUDED.parquet_path,
                          source_file=EXCLUDED.source_file,
                          source_checksum=EXCLUDED.source_checksum,
                          ingest_version=EXCLUDED.ingest_version,
                          updated_at=now()
                """), {
                    "pid": str(profile_id),
                    "fid": profile.get("float_id"),
                    "cyc": profile.get("cycle_number"),
                    "date": profile.get("profile_date"),
                    "lat": lat,
                    "lon": lon,
                    "geom": f"POINT({lon} {lat})",
                    "n_levels": len([x for x in profile.get("PRES",[]) if np.isfinite(x)]),
                    "min_pres": float(np.nanmin(profile.get("PRES",[np.nan]))) if profile.get("PRES") else None,
                    "max_pres": float(np.nanmax(profile.get("PRES",[np.nan]))) if profile.get("PRES") else None,
                    "qc": json.dumps(profile.get("qc_summary",{}),default=convert_numpy),
                    "summary": profile.get("summary_text"),
                    "parquet": parquet_uri,
                    "src": source_file,
                    "cs": checksum,
                    "ver": INGEST_VERSION
                })
            processed += 1

        with engine.begin() as conn:
            conn.execute(text("UPDATE process_log SET status='OK', records_parsed=:n, ended_at=now() WHERE id=:id"),
                         {"n": processed, "id": log_id})
        logger.success(f"🎉 Ingested {source_file}: {processed} profiles processed.")

    except Exception as exc:
        logger.exception(f"❌ Error ingesting {nc_file}: {exc}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import sys, glob
    try:
        ensure_bucket_exists(BUCKET_NAME)
    except Exception:
        logger.warning("Bucket creation failed; continuing.")
    try:
        ensure_parquet_prefix(PARQUET_BASE_URI)
    except Exception:
        logger.debug("Failed ensuring parquet prefix.")
    if len(sys.argv)<2:
        logger.error("Usage: python ingest.py <netcdf_file_or_folder>")
        sys.exit(1)
    target = sys.argv[1]
    files=[]
    if os.path.isdir(target):
        files=glob.glob(os.path.join(target,"*.nc"))
    else:
        files=[target]
    logger.info(f"Found {len(files)} files")
    for f in files:
        ingest_one_file(f)
