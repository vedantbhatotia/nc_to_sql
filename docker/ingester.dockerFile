# docker/ingester.Dockerfile
FROM python:3.10-slim

LABEL maintainer="you@example.com"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# system deps for netcdf/psycopg2/pyarrow
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc libpq-dev libnetcdf-dev wget curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# copy requirements and install once at build time
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# copy project code
COPY . /app

# create non-root user
RUN useradd --create-home --shell /bin/bash appuser \
 && chown -R appuser:appuser /app

USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

# default: idle container; we run ingestion with `docker compose run` or `exec`
CMD ["bash", "-lc", "tail -f /dev/null"]
