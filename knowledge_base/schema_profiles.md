# Profiles & Process Log Schema

This document details the schema for the tables that store data related to individual profiles and the ingestion process itself.

---

## Table: `profiles`

Stores metadata for each individual measurement profile. This is the main table for querying profile locations, times, and data quality.

| Column            | Type          | Description                                                                                             |
|-------------------|---------------|---------------------------------------------------------------------------------------------------------|
| `profile_id`      | `UUID`        | **Primary Key.** A unique identifier for this specific profile record.                                  |
| `float_id`        | `TEXT`        | **Foreign Key** to `floats.float_id`. Links the profile to its parent float.                            |
| `cycle_number`    | `INTEGER`     | The cycle number of this profile for the given float.                                                     |
| `profile_date`    | `TIMESTAMPTZ` | The timestamp when the profile was taken.                                                                 |
| `lat`             | `DOUBLE`      | Latitude of the profile location.                                                                       |
| `lon`             | `DOUBLE`      | Longitude of the profile location.                                                                      |
| `geom`            | `GEOGRAPHY`   | A PostGIS `GEOGRAPHY(Point, 4326)` type for efficient spatial queries. Automatically generated from `lat`/`lon`. |
| `n_levels`        | `INTEGER`     | The number of valid pressure levels in the profile data.                                                |
| `min_pres`        | `DOUBLE`      | The minimum pressure (closest to the surface) recorded in the profile.                                  |
| `max_pres`        | `DOUBLE`      | The maximum pressure (deepest) recorded in the profile.                                                 |
| `qc_summary`      | `JSONB`       | A JSON object summarizing the percentage of "good" data for each variable (e.g., `{"TEMP": 95, "PSAL": 97}`). |
| `summary_text`    | `TEXT`        | A human-readable summary of the profile, suitable for semantic search.                                  |
| `parquet_path`    | `TEXT`        | The S3 path to the Parquet file containing the detailed profile measurements (e.g., `s3://.../file.parquet`). |
| `source_file`     | `TEXT`        | The name of the original NetCDF file this profile was extracted from.                                     |
| `source_checksum` | `TEXT`        | The SHA256 checksum of the source NetCDF file, used for idempotency checks.                               |
| `ingest_version`  | `TEXT`        | The version of the ingestion script that processed this profile.                                          |
| `created_at`      | `TIMESTAMPTZ` | Timestamp when this profile was first ingested.                                                           |
| `updated_at`      | `TIMESTAMPTZ` | Timestamp when this profile record was last updated.                                                      |

**Unique Constraint:** `(float_id, cycle_number)` ensures that each cycle for a given float is only ingested once.

**Triggers:**
- `trg_profiles_geom`: Automatically populates the `geom` column from `lat` and `lon` on insert or update.
- `trg_profiles_touch`: Automatically updates the `updated_at` column on any row modification.

---

## Table: `process_log`

Tracks the status of each file ingestion attempt. This is useful for monitoring and debugging the ingestion pipeline.

| Column           | Type          | Description                                                                 |
|------------------|---------------|-----------------------------------------------------------------------------|
| `id`             | `SERIAL`      | **Primary Key.** A unique identifier for the log entry.                     |
| `file_name`      | `TEXT`        | The name of the NetCDF file being processed.                                |
| `checksum`       | `TEXT`        | The SHA256 checksum of the file.                                            |
| `status`         | `TEXT`        | The outcome of the ingestion: 'OK', 'ERROR', or 'SKIPPED'.                  |
| `error_msg`      | `TEXT`        | The error message if the status is 'ERROR'.                                 |
| `records_parsed` | `INTEGER`     | The number of profiles successfully parsed from the file.                   |
| `started_at`     | `TIMESTAMPTZ` | Timestamp when the file processing started.                                   |
| `ended_at`       | `TIMESTAMPTZ` | Timestamp when the file processing finished.                                  |