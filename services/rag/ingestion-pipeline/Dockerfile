# ── VocalMind Ingestion Pipeline ──────────────────────────────────────────────
# Docling (AI-powered PDF parsing) + Ollama embeddings + Qdrant vector store.
# First build takes 5-10 min — Docling downloads PyTorch CPU + AI model weights.
FROM python:3.11-slim

WORKDIR /app

# OS dependencies required by Docling / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from the parent pyproject.toml [ingestion] extras
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[ingestion]"

# Pre-download Docling AI models into the image (avoids download at runtime)
RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"

# Copy pipeline script
COPY ingestion-pipeline/ingestion_pipeline.py .

# Directories mapped via Docker volumes at runtime
RUN mkdir -p /app/NA_docs /app/parsed_docs

CMD ["python", "ingestion_pipeline.py"]
