# ── VocalMind Final RAG — Ingestion Container ────────────────────────────────
# Docling (AI-powered PDF → Markdown) + Ollama embeddings + Qdrant vector store.
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

# Install Python dependencies
COPY pyproject.toml .
ENV PIP_DEFAULT_TIMEOUT=300
RUN pip install --no-cache-dir --retries 10 .

# Pre-download Docling AI models into the image (avoids download at runtime)
RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"

# Copy all application code
COPY config.py ingest.py query_engine.py evaluator.py main.py __init__.py ./

# Directories used at runtime
RUN mkdir -p /sop-standards /app/logs /app/reports

# Default: keep the ingestion worker alive and refresh on changes.
CMD ["python", "main.py", "--watch"]
