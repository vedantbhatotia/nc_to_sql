Argo NetCDF Ingestion System 🚀

Ingest Argo NetCDF files → PostgreSQL (PostGIS) + Parquet (MinIO S3) with one command.
Includes pgAdmin for browsing profiles.

⚡ Quick Start
# 1. Clone repo
git clone 

# 2. Start infra (Postgres + MinIO + pgAdmin)
docker compose up -d

# 3. Ingest sample NetCDFs
docker compose run --rm ingester python scripts/ingest.py samples/


✅ Profiles appear in PostgreSQL
✅ Parquet files land in MinIO (partitioned by year/month)

📦 Prerequisites

Docker ≥ 20.10

Docker Compose ≥ 1.29

A .env file with credentials:

# PostgreSQL
PG_HOST=postgres
PG_PORT=5432
PG_USER=argo
PG_PASSWORD=argo_pass
PG_DB=argo_db

# MinIO
MINIO_HOST=minio
MINIO_PORT=9000
MINIO_ACCESS_KEY=minio
MINIO_SECRET_KEY=minio123
PARQUET_BASE_URI=s3://argo-data/parquet

# Other
INGEST_VERSION=v1
QUARANTINE_DIR=quarantine


Place .nc files in samples/ for testing.

🔧 Services

PostgreSQL + PostGIS → stores metadata

MinIO → S3-compatible Parquet storage

pgAdmin → PostgreSQL UI (http://localhost:5050
, user: admin@admin.com, pass: admin)

Ingester → Python service that parses .nc and writes to DB + MinIO

📥 Ingestion
docker compose run --rm ingester python scripts/ingest.py samples/


Extracts PRES, TEMP, PSAL, DOXY, CHLA

Computes QC summaries

Inserts metadata into PostgreSQL

Writes Parquet:

s3://argo-data/parquet/year=YYYY/month=MM/<float>_cycle_<n>.parquet


Moves problematic files to quarantine/

🔍 Validation
Parquet (in MinIO)
docker compose exec minio mc alias set local http://localhost:9000 minio minio123
docker compose exec minio mc ls local/argo-data/parquet/year=2025/month=09/

PostgreSQL (via pgAdmin)

URL: http://localhost:5050

User: admin@admin.com

Pass: admin

Connect using .env credentials

Sample queries:

-- All floats
SELECT * FROM floats;

-- Profiles with QC summary
SELECT float_id, cycle_number, profile_date, qc_summary FROM profiles;

-- Count profiles per float
SELECT float_id, COUNT(*) FROM profiles GROUP BY float_id;

-- High-quality temperature profiles
SELECT float_id, cycle_number, profile_date
FROM profiles
WHERE (qc_summary->>'TEMP')::int >= 90;

🎯 End-to-End Demo
docker compose up -d
docker compose run --rm ingester python scripts/ingest.py samples/
docker compose logs ingester


Then:

Browse Parquet in MinIO (or download via S3 tools)

Explore metadata in PostgreSQL via pgAdmin

📝 Notes

Idempotent: duplicate files skipped

Quarantine: problematic NetCDFs stored in quarantine/

Partitioning: Parquet stored by year/month

Customize S3 bucket via PARQUET_BASE_URI