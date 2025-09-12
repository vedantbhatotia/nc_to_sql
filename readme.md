Argo NetCDF Ingestion System 🚀

This project ingests Argo NetCDF files into PostgreSQL + PostGIS, writes per-profile Parquet files to MinIO, and provides pgAdmin for exploration.

1️⃣ Prerequisites

Docker ≥ 20.10

Docker Compose ≥ 1.29

.env file with PostgreSQL and MinIO credentials, e.g.:

PG_HOST=postgres
PG_PORT=5432
PG_USER=argo
PG_PASSWORD=argo_pass
PG_DB=argo_db

MINIO_HOST=minio
MINIO_PORT=9000
MINIO_ACCESS_KEY=minio
MINIO_SECRET_KEY=minio123
PARQUET_BASE_URI=s3://argo-data/parquet

INGEST_VERSION=v1
QUARANTINE_DIR=quarantine


A samples/ folder with NetCDF .nc files for testing.

2️⃣ Services (via Docker Compose)

PostgreSQL + PostGIS: stores float/profile metadata.

MinIO: S3-compatible storage for Parquet files.

pgAdmin: browser-based PostgreSQL UI.

Ingester: Python container running the ingestion script.

Start services
docker compose up -d


Check status:

docker compose ps

3️⃣ Ingestion

Run the ingester on a folder of NetCDF files:

docker compose run --rm ingester python scripts/ingest.py samples/


Each NetCDF file is read and profiles are extracted.

Parquet files are written to s3://argo-data/parquet/year=YYYY/month=MM/....

Metadata is inserted/updated in PostgreSQL.

Problematic files are moved to quarantine/.

4️⃣ Validation
Parquet

You can inspect Parquet files inside MinIO:

docker compose exec minio mc alias set local http://localhost:9000 minio minio123
docker compose exec minio mc ls local/argo-data/parquet/year=2025/month=09/


Or download for local inspection:

docker compose exec ingester bash
aws --endpoint-url=http://minio:9000 s3 cp s3://argo-data/parquet/year=2025/month=09/ ./ -r

PostgreSQL

Access via pgAdmin:

URL: http://localhost:5050

User: admin@admin.com / admin

Add server with .env credentials.

Sample queries:

-- Show all floats
SELECT * FROM floats;

-- Show profiles with QC summary
SELECT float_id, cycle_number, profile_date, qc_summary
FROM profiles
ORDER BY profile_date DESC;

-- Count profiles per float
SELECT float_id, COUNT(*) AS n_profiles
FROM profiles
GROUP BY float_id
ORDER BY n_profiles DESC;

-- Filter by date & QC
SELECT float_id, cycle_number, profile_date
FROM profiles
WHERE profile_date >= '2025-01-01' 
  AND (qc_summary->>'TEMP')::int >= 90;

5️⃣ End-to-End Test 🎯

Start Docker services:

docker compose up -d


Run ingestion:

docker compose run --rm ingester python scripts/ingest.py samples/


Check logs:

docker compose logs ingester


Validate:

Parquet files in MinIO (mc ls or via S3 browser).

Profiles in PostgreSQL (psql or pgAdmin).

Sample query:

SELECT float_id, cycle_number, profile_date FROM profiles LIMIT 10;


✅ Everything should be visible and queryable. Profiles should include TEMP, PSAL, DOXY, CHLA, and QC summary.

6️⃣ Notes / Tips

quarantine/ stores problematic NetCDF files for inspection.

Ingestion is idempotent: duplicate files are skipped.

Parquet files are partitioned by year/month for efficient S3 querying.

Customize PARQUET_BASE_URI in .env to use a different S3 bucket/prefix.

This README ensures teammates or judges can reproduce the full ingestion pipeline quickly, inspect data, and validate both Parquet + PostgreSQL.