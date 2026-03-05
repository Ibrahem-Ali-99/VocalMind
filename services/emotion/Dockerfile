FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# explicitly define where the model should be downloaded within the container
ENV MODELSCOPE_CACHE=/app/model_cache
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python if not already done
RUN ln -s /usr/bin/python3 /usr/bin/python || true

# Install python dependencies using pip for maximum compatibility
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Download model inside Docker image to cache weights
COPY download_model.py .
RUN python download_model.py

# Copy application code
COPY app.py .

# Expose API port
EXPOSE 8000

# Start server
CMD ["python", "app.py"]
