# Floats Table Schema

The `floats` table stores static, global metadata for each unique Argo float. This table is populated or updated whenever a new float is encountered during ingestion.

## Columns

| Column          | Type          | Description                                                                 |
|-----------------|---------------|-----------------------------------------------------------------------------|
| `float_id`      | `TEXT`        | **Primary Key.** The unique identifier for the float (e.g., '1901290').      |
| `platform_type` | `TEXT`        | The type of the Argo platform (e.g., 'PROVOR').                             |
| `wmo_id`        | `TEXT`        | The World Meteorological Organization ID for the float.                       |
| `launch_date`   | `DATE`        | The date the float was launched.                                            |
| `home_center`   | `TEXT`        | The data center or institution responsible for the float.                     |
| `sensor_types`  | `TEXT[]`      | An array of sensor types on the float (e.g., `{'CTD', 'BGC'}`).             |
| `metadata`      | `JSONB`       | A JSON blob containing all other global attributes from the NetCDF file.      |
| `created_at`    | `TIMESTAMPTZ` | Timestamp of the first time this float was seen.                              |
| `updated_at`    | `TIMESTAMPTZ` | Timestamp of the last time this float's metadata was updated.                 |

## Indexes

- `idx_floats_metadata_gin`: A GIN index on the `metadata` JSONB column to accelerate queries into its content.
- `idx_floats_sensors_gin`: A GIN index on the `sensor_types` array to accelerate queries filtering by sensor type.