FROM python:3.10-slim

LABEL maintainer="you@example.com"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc libpq-dev wget curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# This is its own layer, so it's only rebuilt if requirements.txt changes
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy only the necessary application code
COPY scripts/ /app/scripts/
COPY knowledge_base/ /app/knowledge_base/

# Create a non-root user and take ownership of the app directory
RUN useradd --create-home --shell /bin/bash appuser \
 && chown -R appuser:appuser /app

USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"
