# ============================================================
# prompt-hunter — Dockerfile
# ============================================================
# Multi-stage build for minimal final image with GPU support.
#
# Build:
#   docker build -t prompt-hunter .
#
# Run:
#   docker run --gpus all -v /path/to/data:/data prompt-hunter \
#       --class forklift --train-json /data/train.json --train-images /data/train/
# ============================================================

# ---------- Stage 1: Builder ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for opencv (headless)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim

WORKDIR /app

# OpenCV runtime deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/prompt-hunter /usr/local/bin/prompt-hunter

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

ENTRYPOINT ["prompt-hunter"]
CMD ["--help"]
