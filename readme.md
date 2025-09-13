# Argo NetCDF Ingestion System 

Ingest **Argo NetCDF** files into **PostgreSQL (PostGIS)** and **Parquet (MinIO S3)** with a single command. The system also includes **pgAdmin** for browsing profiles.

-----

### Quick Start

1.  **Clone the repository**

    ```bash
    git clone <repo-url>
    cd <repo-folder>
    ```

2.  **Setup environment variables**
    Copy the example `.env` and edit it if necessary:

    ```bash
    cp .env.example .env
    ```

    **Example `.env`:**

    ```
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
    ```

3.  **Start infrastructure**
    Start PostgreSQL, MinIO, and pgAdmin:

    ```bash
    docker compose up -d
    ```

    *Note: PostgreSQL automatically initializes the schema from `./db-init/schema.sql`.*

    Wait until Postgres is ready:

    ```bash
    docker compose logs -f postgres
    # look for "database system is ready to accept connections"
    ```

4.  **Ingest sample NetCDF files**
    Place your `.nc` files in the `samples/` folder, then run:

    ```bash
    docker compose run --rm ingester python scripts/ingest.py samples/
    ```

    **What happens:**

      - Profiles extracted from NetCDF
      - Metadata inserted into PostgreSQL
      - Parquet files written to MinIO (partitioned by year/month)
      - Problematic files moved to `quarantine/`

    *Notes:*

      - **Idempotent:** duplicate files are skipped automatically.
      - To re-ingest the same profile, delete existing rows first:

    <!-- end list -->

    ```sql
    DELETE FROM profiles
    WHERE float_id='1901290' AND cycle_number=249;
    ```

5.  **Validate ingestion**

    **PostgreSQL (via pgAdmin)**

      - **URL:** `http://localhost:5050`
      - **User:** `admin@admin.com`
      - **Password:** `admin`
      - Connect to server: `postgres` (host), port `5432`, credentials from `.env`

    **Sample queries:**

    ```sql
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
    ```

    **MinIO (S3 storage)**

      - **Web UI:** `http://localhost:9000` or `http://localhost:9001`
      - **Credentials:** `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY`

    **Example CLI:**

    ```bash
    # Alias MinIO locally
    docker compose exec minio mc alias set local http://localhost:9000 minio minio123

    # List Parquet files
    docker compose exec minio mc ls local/argo-data/parquet/year=2025/month=09/
    ```

6.  **End-to-End Demo**

    ```bash
    # Start services
    docker compose up -d

    # Ingest files
    docker compose run --rm ingester python scripts/ingest.py samples/

    # View ingester logs
    docker compose logs ingester
    ```

7.  **Notes / Tips**

      - **Idempotency:** Duplicate files are skipped; ingestion is checked via checksum.
      - **Quarantine:** Problematic NetCDFs are moved to `quarantine/` for inspection.
      - **Partitioning:** Parquet files are stored by `year/month`.
      - **Force re-ingestion:** Delete rows from the `profiles` table for a specific `float_id`/`cycle_number`.
      - **Custom S3 bucket:** Update `PARQUET_BASE_URI` in `.env`.

8.  **Prerequisites**

      - Docker ≥ 20.10
      - Docker Compose ≥ 1.29
      - Optional: MinIO Client (`mc`) for local S3 management
      - `.env` file with credentials
