# PostGIS Query Examples

The `profiles` table contains a `geom` column of type `GEOGRAPHY(Point, 4326)`. This allows for powerful and efficient geospatial queries using PostGIS functions. The `GEOGRAPHY` type correctly handles calculations on a spheroid (the Earth's shape), making distance calculations accurate.

A GIST (Generalized Search Tree) index named `idx_profiles_geom` is created on this column to ensure these queries are fast.

---

### Example 1: Find profiles within a radius

Find all profiles within 500 kilometers of a specific point (e.g., near Hawaii at 21°N, 157°W). The `ST_DWithin` function uses the spatial index for excellent performance.

```sql
SELECT
  float_id,
  cycle_number,
  profile_date
FROM profiles
WHERE ST_DWithin(
  geom,
  ST_MakePoint(-157.8583, 21.3069)::geography,
  500000  -- Distance in meters
);
```

### Example 2: Find profiles within a bounding box

Find all profiles within a rectangular area covering the North Atlantic. The `&&` operator (intersects) will also use the spatial index.

```sql
SELECT float_id, cycle_number, profile_date, lat, lon
FROM profiles
WHERE geom && ST_MakeEnvelope(-80, 20, -10, 65, 4326);
```