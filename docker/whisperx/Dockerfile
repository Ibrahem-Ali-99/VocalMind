FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=agg
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
ENV LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set python3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

WORKDIR /app

# Create venv and install PyTorch first (must match CUDA 12.8)
ENV UV_HTTP_TIMEOUT=300
RUN uv venv .venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -p .venv/bin/python \
    torch==2.10.0 \
    torchaudio==2.10.0 \
    torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install WhisperX and core dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -p .venv/bin/python \
    whisperx \
    transformers==5.0.0 \
    tokenizers==0.22.2 \
    huggingface-hub==1.4.1 \
    accelerate==1.12.0 \
    safetensors==0.7.0 \
    soundfile==0.13.1 \
    librosa==0.11.0 \
    numpy==2.0.2 \
    pandas==2.2.2 \
    tqdm==4.67.3

# Activate venv by default
ENV PATH="/app/.venv/bin:$PATH"

# Create input/output directories
RUN mkdir -p /app/audio /app/output

WORKDIR /app

# Default: transcribe any .mp3/.wav placed in /app/audio
# Override CMD at runtime with your own flags
CMD ["whisperx", "--help"]
