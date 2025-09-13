-- ==============================
-- Enable Extensions
-- ==============================
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ==============================
-- Table: floats (static float metadata)
-- ==============================
CREATE TABLE IF NOT EXISTS floats (
  float_id TEXT PRIMARY KEY,
  platform_type TEXT,
  wmo_id TEXT,
  launch_date DATE,
  home_center TEXT,
  sensor_types TEXT[],       -- e.g., {'CTD','BGC'}
  metadata JSONB,            -- extra global attributes
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Ensure all columns exist if schema evolves
ALTER TABLE floats
ADD COLUMN IF NOT EXISTS platform_type TEXT,
ADD COLUMN IF NOT EXISTS wmo_id TEXT,
ADD COLUMN IF NOT EXISTS launch_date DATE,
ADD COLUMN IF NOT EXISTS home_center TEXT,
ADD COLUMN IF NOT EXISTS sensor_types TEXT[],
ADD COLUMN IF NOT EXISTS metadata JSONB;

-- ==============================
-- Table: profiles (one row per profile)
-- ==============================
CREATE TABLE IF NOT EXISTS profiles (
  profile_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  float_id TEXT REFERENCES floats(float_id) ON DELETE SET NULL,
  cycle_number INTEGER,
  profile_date TIMESTAMPTZ,
  lat DOUBLE PRECISION,
  lon DOUBLE PRECISION,
  geom GEOGRAPHY(Point, 4326),    -- for spatial queries
  n_levels INTEGER,
  min_pres DOUBLE PRECISION,
  max_pres DOUBLE PRECISION,
  qc_summary JSONB,               -- e.g., {"TEMP":95,"PSAL":97}
  summary_text TEXT,
  parquet_path TEXT,              -- pointer to Parquet row/file
  source_file TEXT,
  source_checksum TEXT,
  ingest_version TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (float_id, cycle_number)
);

-- Ensure constraints for lat/lon
ALTER TABLE profiles
ADD CONSTRAINT IF NOT EXISTS chk_lat_range CHECK (lat IS NULL OR (lat >= -90 AND lat <= 90)),
ADD CONSTRAINT IF NOT EXISTS chk_lon_range CHECK (lon IS NULL OR (lon >= -180 AND lon <= 180));

-- ==============================
-- Table: process_log (ingestion tracking)
-- ==============================
CREATE TABLE IF NOT EXISTS process_log (
  id SERIAL PRIMARY KEY,
  file_name TEXT,
  checksum TEXT,
  status TEXT, -- 'OK','ERROR','SKIPPED'
  error_msg TEXT,
  records_parsed INTEGER,
  started_at TIMESTAMPTZ DEFAULT now(),
  ended_at TIMESTAMPTZ
);

-- ==============================
-- Indexes
-- ==============================
CREATE INDEX IF NOT EXISTS idx_profiles_date ON profiles(profile_date);
CREATE INDEX IF NOT EXISTS idx_profiles_float ON profiles(float_id);
CREATE INDEX IF NOT EXISTS idx_profiles_cycle ON profiles(cycle_number);
CREATE INDEX IF NOT EXISTS idx_profiles_geom ON profiles USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_profiles_float_date ON profiles(float_id, profile_date DESC);

CREATE INDEX IF NOT EXISTS idx_profiles_qcsummary_gin ON profiles USING GIN (qc_summary);
CREATE INDEX IF NOT EXISTS idx_floats_metadata_gin ON floats USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_floats_sensors_gin ON floats USING GIN (sensor_types);

-- ==============================
-- Functions
-- ==============================
-- Auto-update updated_at
CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Auto-populate geom from lat/lon
CREATE OR REPLACE FUNCTION latlon_to_geom()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.lat IS NOT NULL AND NEW.lon IS NOT NULL THEN
    NEW.geom = ST_SetSRID(ST_MakePoint(NEW.lon, NEW.lat), 4326)::geography;
  ELSE
    NEW.geom = NULL;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ==============================
-- Triggers
-- ==============================
-- Drop existing triggers if they exist
DROP TRIGGER IF EXISTS trg_profiles_touch ON profiles;
CREATE TRIGGER trg_profiles_touch
BEFORE INSERT OR UPDATE ON profiles
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS trg_profiles_geom ON profiles;
CREATE TRIGGER trg_profiles_geom
BEFORE INSERT OR UPDATE ON profiles
FOR EACH ROW EXECUTE FUNCTION latlon_to_geom();
