# ──────────────────────────────────────────────────────────────────────────────
# WhisperX Service — Docker Image
# ──────────────────────────────────────────────────────────────────────────────
# Build:   docker build -t whisperx-service .
# Run:     docker run --gpus all -p 8000:8000 --env-file ../../../../.env whisperx-service
# ──────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# ── Install uv ───────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ── App setup ─────────────────────────────────────────────────────────────────
WORKDIR /app

# Copy dependency metadata first for better layer caching
COPY pyproject.toml .

# Install dependencies with uv
RUN uv sync

# Copy application code after dependencies are installed
COPY app.py .

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000

# Run with uvicorn
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
