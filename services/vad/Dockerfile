# syntax=docker/dockerfile:1.4
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps (cache pip wheels between builds)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=100 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-download Silero VAD model at build time (Torch Hub cache survives rebuilds)
COPY download_model.py .
RUN --mount=type=cache,target=/root/.cache/torch \
    python download_model.py

# Copy application
COPY app.py .

EXPOSE 8002

CMD ["python", "app.py"]
