"""
VocalMind Universal Ingestion Pipeline — Dockerized with Docling
=================================================================
Dual-Granularity RAG Pipeline.

Architecture (fully local, no external APIs):
  PDF parsing  → Docling        (AI-powered PDF → Markdown, best-in-class table accuracy)
  Chunking     → langchain-text-splitters
  Embeddings   → Ollama         (nomic-embed-text, 768-dim)
  Vector store → Qdrant         (persisted on disk via Docker volume)

Collections in Qdrant:
  vocalmind_parents   — full policy sections  (compliance evaluation)
  vocalmind_children  — precision snippets    (fact retrieval)

Environment variables injected by docker-compose.yml:
  OLLAMA_URL       http://ollama:11434
  QDRANT_URL       http://qdrant:6333
  EMBEDDING_MODEL  nomic-embed-text
"""

import os
import re
import glob
import json
import uuid
import hashlib
from datetime import datetime

import httpx
import ftfy
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHILD_CHUNK_SIZE        = 400
CHILD_CHUNK_OVERLAP     = 80
MIN_CHUNK_LENGTH        = 30
EMPTY_SECTION_MIN_WORDS = 4
EMBEDDING_DIM           = 768   # nomic-embed-text output dimension

COLLECTION_PARENTS  = "vocalmind_parents"
COLLECTION_CHILDREN = "vocalmind_children"

# H1 + H2 + H3 — splits on all major heading levels for granular sections
HEADERS_TO_SPLIT_ON = [
    ("#",   "Header 1"),
    ("##",  "Header 2"),
    ("###", "Header 3"),
]

OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://localhost:11434")
QDRANT_URL      = os.getenv("QDRANT_URL",      "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


# ---------------------------------------------------------------------------
# STEP 1 — PDF → Markdown via Docling
# ---------------------------------------------------------------------------
# Initialise once — Docling loads its AI models at startup (TableFormer,
# DocLayNet). Keeping it as a module-level singleton avoids reloading
# the models for every PDF.
_docling_converter = None

def get_converter() -> DocumentConverter:
    global _docling_converter
    if _docling_converter is None:
        print("  Initialising Docling converter (loading AI models — first call only)...")
        _docling_converter = DocumentConverter()
        print("  ✓ Docling ready.")
    return _docling_converter


def parse_pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert a PDF to clean Markdown using Docling.

    Docling uses:
      - DocLayNet  → layout analysis (reading order, headers, body text)
      - TableFormer → table structure recognition (~98% accuracy on complex tables)

    This is the direct replacement for the LlamaCloud API call — fully local,
    no network required.
    """
    print("  Parsing with Docling...")
    converter  = get_converter()
    result     = converter.convert(pdf_path)
    md_text    = result.document.export_to_markdown()
    print(f"  Parsed {len(md_text):,} characters of Markdown.")
    return md_text


# ---------------------------------------------------------------------------
# Embeddings — Ollama REST API
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    """Request an embedding vector from the local Ollama container."""
    response = httpx.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["embedding"]


# ---------------------------------------------------------------------------
# Qdrant — collection setup
# ---------------------------------------------------------------------------

def ensure_collections(client: QdrantClient):
    """Create Qdrant collections if they don't already exist."""
    existing = [c.name for c in client.get_collections().collections]
    for name in [COLLECTION_PARENTS, COLLECTION_CHILDREN]:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            print(f"  ✓ Created collection: {name}")
        else:
            print(f"  Collection already exists: {name}")


# ---------------------------------------------------------------------------
# Qdrant — upload chunks
# ---------------------------------------------------------------------------

def upload_chunks_to_qdrant(
    client: QdrantClient,
    chunks: list,
    collection_name: str,
    label: str,
):
    """
    Embed each chunk via Ollama and upsert it into Qdrant.

    Point structure:
      id      — UUID derived from content MD5 hash (deterministic, safe to re-run)
      vector  — 768-dim embedding from nomic-embed-text
      payload — chunk text + all metadata (org, doc_id, effective_date, etc.)
    """
    points = []
    print(f"  Embedding {len(chunks)} {label} chunks...")

    for i, chunk in enumerate(chunks):
        content = chunk.page_content.strip()
        if not content:
            continue

        # Deterministic UUID — same content always gets the same ID (upsert-safe)
        content_hash = hashlib.md5(content.encode()).hexdigest()
        point_id     = str(uuid.UUID(content_hash))

        vector  = get_embedding(content)
        payload = {"text": content}
        payload.update(chunk.metadata)

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"    {i+1}/{len(chunks)} embedded...")

    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"  ✓ Uploaded {len(points)} points → '{collection_name}'")


# ---------------------------------------------------------------------------
# Helpers — encoding, metadata, table repair, validation
# ---------------------------------------------------------------------------

def fix_encoding(text: str) -> str:
    text = ftfy.fix_text(text)
    for entity, char in {
        "&#x26;": "&", "&#x3C;": "<", "&#x3E;": ">",
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#x27;": "'",
    }.items():
        text = text.replace(entity, char)
    return text


def extract_doc_metadata(markdown_text: str, source_file: str) -> dict:
    """Handles both plain and **bold** Markdown field formatting."""
    def make_pattern(label: str) -> str:
        return (
            r"^\*{0,2}" + re.escape(label) + r"\*{0,2}"
            + r"\s*:\s*\*{0,2}(.+?)\*{0,2}\s*$"
        )
    fields = {
        "org":            make_pattern("Organization"),
        "department":     make_pattern("Department"),
        "doc_id":         make_pattern("Document ID"),
        "version":        make_pattern("Version"),
        "effective_date": make_pattern("Effective Date"),
    }
    extracted = {}
    for key, pattern in fields.items():
        m = re.search(pattern, markdown_text, re.IGNORECASE | re.MULTILINE)
        extracted[key] = m.group(1).strip() if m else "Unknown"
    extracted["source_file"] = source_file
    extracted["ingested_at"] = datetime.utcnow().isoformat() + "Z"
    return extracted


def repair_orphaned_table_rows(markdown_text: str) -> str:
    """Re-attach pipe-table rows separated from their table by blank lines."""
    lines, output, i = markdown_text.splitlines(), [], 0
    while i < len(lines):
        output.append(lines[i])
        if lines[i].strip().startswith("|"):
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and lines[j].strip().startswith("|"):
                i = j
                continue
        i += 1
    return "\n".join(output)


def is_table_chunk(text: str) -> bool:
    """Detect Markdown pipe tables OR raw HTML <table> blocks."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if sum(1 for ln in lines if ln.startswith("|")) >= 2:
        return True
    if re.search(r"<table[\s>]", text, re.IGNORECASE):
        return True
    return False


def is_empty_section(chunk) -> bool:
    content = re.sub(r"[#\*\-\_`\[\]\(\)]", " ", chunk.page_content)
    content = re.sub(r"\s+", " ", content).strip()
    return len([w for w in content.split() if w]) < EMPTY_SECTION_MIN_WORDS


def validate_chunks(chunks: list, label: str) -> dict:
    warnings, seen, short_count, dupe_count = [], {}, 0, 0
    for i, chunk in enumerate(chunks):
        content = chunk.page_content.strip()
        if len(content) < MIN_CHUNK_LENGTH:
            short_count += 1
            warnings.append(f"[{label}] Chunk {i+1} too short ({len(content)} chars)")
        h = hashlib.md5(content.encode()).hexdigest()
        if h in seen:
            dupe_count += 1
            warnings.append(f"[{label}] Chunk {i+1} duplicates chunk {seen[h]+1}")
        else:
            seen[h] = i
    return {"total": len(chunks), "short": short_count,
            "duplicates": dupe_count, "warnings": warnings}


def split_into_children(parent_chunks: list, child_splitter) -> list:
    child_chunks, atomic_count = [], 0
    for chunk in parent_chunks:
        if is_table_chunk(chunk.page_content):
            chunk.metadata["chunk_type"] = "table_atomic"
            child_chunks.append(chunk)
            atomic_count += 1
        else:
            chunk.metadata["chunk_type"] = "text"
            child_chunks.extend(child_splitter.split_documents([chunk]))
    print(f"  → {atomic_count} table chunk(s) kept atomic.")
    return child_chunks


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------

def process_file(pdf_file: str, output_dir: str, qdrant_client: QdrantClient) -> dict | None:
    base_name = os.path.splitext(os.path.basename(pdf_file))[0]
    print(f"\n{'='*70}")
    print(f"Processing: {os.path.basename(pdf_file)}")
    print(f"{'='*70}")

    # STEP 1 — Parse PDF with Docling
    print("\n[Step 1] Parsing PDF with Docling...")
    try:
        raw_markdown = parse_pdf_to_markdown(pdf_file)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    raw_path = os.path.join(output_dir, f"{base_name}_raw.md")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_markdown)
    print(f"  Saved raw markdown → {raw_path}")

    # STEP 2 — Encoding fix
    print("\n[Step 2] Fixing encoding artifacts...")
    clean_markdown = fix_encoding(raw_markdown)

    # STEP 2b — Repair orphaned table rows
    print("\n[Step 2b] Repairing orphaned table rows...")
    clean_markdown = repair_orphaned_table_rows(clean_markdown)

    clean_path = os.path.join(output_dir, f"{base_name}.md")
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(clean_markdown)
    print(f"  Saved clean markdown → {clean_path}")

    # STEP 3 — Extract metadata
    print("\n[Step 3] Extracting document metadata...")
    doc_meta = extract_doc_metadata(clean_markdown, base_name)
    print(f"  {json.dumps({k: v for k, v in doc_meta.items() if k != 'ingested_at'})}")

    # STEP 4 — Parent splitting (H1 / H2 / H3)
    print("\n[Step 4] Splitting into Parent Chunks (H1 / H2 / H3)...")
    md_splitter       = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    all_parents       = md_splitter.split_text(clean_markdown)
    empty_sections    = []
    parent_chunks     = []
    HEADER_KEYS       = {"Header 1", "Header 2", "Header 3"}

    for chunk in all_parents:
        if is_empty_section(chunk):
            parts = [v for k, v in chunk.metadata.items() if k in HEADER_KEYS]
            name  = " > ".join(parts) if parts else "Unknown"
            empty_sections.append(name)
            print(f"  ⚠  Empty section skipped: [{name}]")
        else:
            chunk.metadata.update(doc_meta)
            parent_chunks.append(chunk)

    print(f"  Parent chunks: {len(parent_chunks)}")

    # STEP 5 — Child splitting
    print("\n[Step 5] Splitting into Child Chunks...")
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    child_chunks = split_into_children(parent_chunks, child_splitter)
    print(f"  Child chunks: {len(child_chunks)}")

    # STEP 6 — Validation
    print("\n[Step 6] Validating...")
    p_report     = validate_chunks(parent_chunks, "PARENT")
    c_report     = validate_chunks(child_chunks,  "CHILD")
    all_warnings = p_report["warnings"] + c_report["warnings"]
    if c_report["total"] == p_report["total"] and len(parent_chunks) > 0:
        all_warnings.append("[PIPELINE] Parent == Child count — child splitting may not be working.")
    if all_warnings:
        for w in all_warnings:
            print(f"  ⚠  {w}")
    else:
        print("  ✓ All chunks passed quality checks.")

    # STEP 7 — Save debug outputs
    print("\n[Step 7] Saving debug outputs...")
    chunks_path = os.path.join(output_dir, f"{base_name}_chunks.md")
    with open(chunks_path, "w", encoding="utf-8") as f:
        f.write(f"# Chunk Analysis: {base_name}\n\n")
        f.write(f"**Ingested at**: {doc_meta['ingested_at']}  \n")
        f.write(f"**Organization**: {doc_meta['org']}  \n")
        f.write(f"**Document ID**: {doc_meta['doc_id']}  \n\n")
        f.write(f"## Parent Chunks ({len(parent_chunks)})\n\n")
        for i, chunk in enumerate(parent_chunks):
            f.write(f"### Parent Chunk {i+1}\n")
            f.write(f"**Metadata**: {chunk.metadata}\n\n")
            f.write(f"```markdown\n{chunk.page_content}\n```\n\n---\n\n")
        f.write(f"## Child Chunks ({len(child_chunks)})\n\n")
        for i, chunk in enumerate(child_chunks):
            label = " 🔒 [TABLE]" if chunk.metadata.get("chunk_type") == "table_atomic" else ""
            f.write(f"### Child Chunk {i+1}{label}\n")
            f.write(f"**Metadata**: {chunk.metadata}\n\n")
            f.write(f"```markdown\n{chunk.page_content}\n```\n\n---\n\n")
    print(f"  Saved → {chunks_path}")

    validation = {
        "file": base_name, "empty_sections": empty_sections,
        "parent_chunks": p_report, "child_chunks": c_report,
    }
    val_path = os.path.join(output_dir, f"{base_name}_validation.json")
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    print(f"  Saved → {val_path}")

    # STEP 8 — Upload to Qdrant
    print("\n[Step 8] Uploading to Qdrant...")
    upload_chunks_to_qdrant(qdrant_client, parent_chunks, COLLECTION_PARENTS,  "parent")
    upload_chunks_to_qdrant(qdrant_client, child_chunks,  COLLECTION_CHILDREN, "child")

    print(f"\n{'─'*50}")
    print(f"  DONE: {base_name}")
    print(f"  Parents  : {len(parent_chunks)} → {COLLECTION_PARENTS}")
    print(f"  Children : {len(child_chunks)} → {COLLECTION_CHILDREN}")
    print(f"  Warnings : {len(all_warnings)}")
    print(f"{'─'*50}")

    return validation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()

    docs_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NA_docs")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parsed_docs")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(docs_dir):
        print(f"ERROR: '{docs_dir}' not found. Place PDFs inside a 'NA_docs/' folder.")
        return

    # Connect to Qdrant
    print(f"\nConnecting to Qdrant at {QDRANT_URL}...")
    qdrant_client = QdrantClient(url=QDRANT_URL)
    ensure_collections(qdrant_client)

    # Verify Ollama + embedding model
    print(f"\nVerifying Ollama at {OLLAMA_URL}...")
    try:
        dim = len(get_embedding("test connection"))
        print(f"  ✓ Ollama reachable. Embedding dim = {dim}")
        if dim != EMBEDDING_DIM:
            print(f"  ⚠  WARNING: expected dim {EMBEDDING_DIM}, got {dim}.")
            print("     Update EMBEDDING_DIM in the script if using a different model.")
    except Exception as e:
        print(f"  ERROR: Cannot reach Ollama: {e}")
        print("  Make sure Ollama is running and nomic-embed-text is pulled.")
        return

    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in '{docs_dir}'.")
        return

    print(f"\nFound {len(pdf_files)} PDF(s). Starting pipeline...\n")

    summaries = []
    for pdf_file in sorted(pdf_files):
        result = process_file(pdf_file, output_dir, qdrant_client)
        if result:
            summaries.append(result)

    report = {
        "run_at":          datetime.utcnow().isoformat() + "Z",
        "files_processed": len(summaries),
        "qdrant_url":      QDRANT_URL,
        "embedding_model": EMBEDDING_MODEL,
        "collections":     {"parents": COLLECTION_PARENTS, "children": COLLECTION_CHILDREN},
        "summaries":       summaries,
    }
    report_path = os.path.join(output_dir, "_pipeline_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print("Pipeline complete!")
    print(f"  Report       → {report_path}")
    print(f"  Qdrant UI    → {QDRANT_URL}/dashboard")
    print(f"  Parents      → {QDRANT_URL}/collections/{COLLECTION_PARENTS}")
    print(f"  Children     → {QDRANT_URL}/collections/{COLLECTION_CHILDREN}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
