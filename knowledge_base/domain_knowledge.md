# Domain Knowledge

This document provides high-level context about the Argo project and its data.

## What is Argo?

Argo is a global array of thousands of robotic profiling floats that measure temperature, salinity, and other properties of the upper ocean. This international program provides a continuous and real-time stream of data that is crucial for climate science, oceanography, and weather forecasting.

## What is a Profile?

A "profile" refers to a single vertical set of measurements taken as a float ascends from a depth (typically 2000 meters) to the surface. Each profile is associated with a specific time and location (`lat`, `lon`). A single float will complete many of these profiling "cycles" over its multi-year lifespan.

## Data Formats

- **NetCDF (Network Common Data Form):** This is the scientific data format in which the source Argo data is distributed. It's a self-describing, machine-independent format for array-oriented data. Our ingestion script reads these `.nc` files.

- **Apache Parquet:** This is a columnar storage format optimized for big data processing. We store the detailed, high-resolution measurement data (pressure, temperature, etc.) for each profile in Parquet files in an S3-compatible object store. This is efficient for analytical queries that only need a subset of columns.

## Key Variables

The core variables measured by Argo floats include:

| Variable | Description                 | Unit          |
|----------|-----------------------------|---------------|
| `PRES`   | Sea Pressure                | decibar (dbar)|
| `TEMP`   | Sea Temperature             | degrees Celsius|
| `PSAL`   | Practical Salinity          | PSS-78 (unitless) |

Many floats also carry Biogeochemical (BGC) sensors, which can measure:

| Variable | Description                 | Unit          |
|----------|-----------------------------|---------------|
| `DOXY`   | Dissolved Oxygen            | µmol/kg       |
| `CHLA`   | Chlorophyll-a               | mg/m³         |