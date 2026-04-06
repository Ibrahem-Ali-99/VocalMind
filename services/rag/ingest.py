"""
VocalMind Final RAG — Document Ingestion Pipeline.

Dual-granularity ingestion:
  Parents  → full policy sections  (compliance evaluation)
  Children → precision snippets    (fact-checking / answer scoring)

Architecture (fully local except Groq LLM):
  PDF parsing  → Docling        (DocLayNet + TableFormer)
  Chunking     → langchain-text-splitters (Markdown header + recursive char)
  Embeddings   → Ollama          (snowflake-arctic-embed2, 1024-dim)
  Vector store → Qdrant          (Docker volume, cosine similarity)
"""

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import ftfy
import httpx
from docling.document_converter import DocumentConverter
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, FilterSelector, MatchValue, PointStruct, VectorParams

try:
    from .config import settings
except ImportError:  # pragma: no cover - allows direct script/test imports
    from config import settings


# ── Docling Singleton ─────────────────────────────────────────────────────────

_docling_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    """Lazily initialise the Docling converter (loads AI models once)."""
    global _docling_converter
    if _docling_converter is None:
        print("  Initialising Docling converter (loading AI models — first call only)...")
        _docling_converter = DocumentConverter()
        print("  ✓ Docling ready.")
    return _docling_converter


# ── Main Pipeline Class ──────────────────────────────────────────────────────


class DocumentIngestionPipeline:
    """
    Ingests PDF policy documents into Qdrant with dual-granularity chunking.

    Usage::

        pipeline = DocumentIngestionPipeline()
        pipeline.run()                       # ingest from configured DOCS_DIR
        pipeline.run(force=True)             # force re-index (wipe + re-ingest)
    """

    def __init__(self) -> None:
        self._connect_qdrant()
        self._verify_ollama()

    # ── Infrastructure setup ──────────────────────────────────────────────

    def _connect_qdrant(self) -> None:
        """Connect to Qdrant and ensure both collections exist."""
        print(f"\nConnecting to Qdrant at {settings.qdrant.url}...")
        self.qdrant = QdrantClient(url=settings.qdrant.url)
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        """Create parent and child collections if they don't exist, or recreate on dimension mismatch."""
        existing = {c.name for c in self.qdrant.get_collections().collections}
        target_dim = settings.embedding.dimension
        for name in [
            settings.qdrant.collection_parents,
            settings.qdrant.collection_children,
            settings.qdrant.collection_sop_parents,
        ]:
            if name in existing:
                # Check existing dimension
                info = self.qdrant.get_collection(name)
                existing_dim = info.config.params.vectors.size
                if existing_dim != target_dim:
                    print(
                        f"  ⚠  Collection '{name}' has dim={existing_dim}, "
                        f"expected {target_dim}. Recreating..."
                    )
                    self.qdrant.delete_collection(name)
                else:
                    print(f"  Collection already exists: {name} (dim={existing_dim})")
                    continue
            self.qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=target_dim,
                    distance=Distance.COSINE,
                ),
            )
            print(f"  ✓ Created collection: {name} (dim={target_dim})")

    def _verify_ollama(self) -> None:
        """Check that Ollama is reachable and the embedding model works."""
        print(f"\nVerifying Ollama at {settings.embedding.base_url}...")
        try:
            vec = self._get_embedding("test connection")
            dim = len(vec)
            print(f"  ✓ Ollama reachable. Embedding dim = {dim}")
            if dim != settings.embedding.dimension:
                print(
                    f"  ⚠  WARNING: expected dim {settings.embedding.dimension}, got {dim}. "
                    f"Update embedding.dimension in config."
                )
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach Ollama: {e}\n"
                f"Make sure Ollama is running and {settings.embedding.model} is pulled:\n"
                f"  docker exec vocalmind_ollama ollama pull {settings.embedding.model}"
            ) from e

    # ── Embedding ─────────────────────────────────────────────────────────

    def _get_embedding(self, text: str) -> list[float]:
        """Request an embedding vector from the local Ollama instance."""
        retry_delays = (0.4, 1.0, 2.0)
        payloads = (
            ("/api/embed", {"model": settings.embedding.model, "input": text}),
            ("/api/embeddings", {"model": settings.embedding.model, "prompt": text}),
        )
        last_error: Exception | None = None
        for attempt, delay in enumerate((0.0, *retry_delays), start=1):
            if delay:
                time.sleep(delay)

            for path, payload in payloads:
                try:
                    response = httpx.post(
                        f"{settings.embedding.base_url}{path}",
                        json=payload,
                        timeout=settings.embedding.request_timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
                    if "embedding" in data:
                        return data["embedding"]
                except Exception as exc:
                    last_error = exc

            if attempt == 1:
                print("  Ollama embedding request failed, retrying...")

        raise ConnectionError(
            f"Cannot reach Ollama embeddings API at {settings.embedding.base_url}: {last_error}"
        )

    # ── PDF Parsing ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_pdf(pdf_path: str) -> str:
        """Convert a PDF to clean Markdown using Docling (DocLayNet + TableFormer)."""
        print("  Parsing with Docling...")
        converter = _get_converter()
        result = converter.convert(pdf_path)
        md_text = result.document.export_to_markdown()
        print(f"  Parsed {len(md_text):,} characters of Markdown.")
        return md_text

    # ── Text Cleaning ─────────────────────────────────────────────────────

    @staticmethod
    def _fix_encoding(text: str) -> str:
        """Fix unicode artifacts and HTML entities."""
        text = ftfy.fix_text(text)
        for entity, char in {
            "&#x26;": "&", "&#x3C;": "<", "&#x3E;": ">",
            "&amp;": "&", "&lt;": "<", "&gt;": ">",
            "&quot;": '"', "&#x27;": "'",
        }.items():
            text = text.replace(entity, char)
        return text

    @staticmethod
    def _repair_orphaned_table_rows(text: str) -> str:
        """Re-attach pipe-table rows separated from their table by blank lines."""
        lines = text.splitlines()
        output: list[str] = []
        i = 0
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

    def _clean_markdown(self, text: str) -> str:
        """Full cleaning pipeline: encoding fix + table repair."""
        text = self._fix_encoding(text)
        text = self._repair_orphaned_table_rows(text)
        return text

    # ── Metadata Extraction ───────────────────────────────────────────────

    @staticmethod
    def _extract_metadata(markdown_text: str, source_file: str, org_name: str) -> dict:
        """Extract structured metadata from Markdown policy header fields."""

        def _pat(label: str) -> str:
            return (
                r"^\*{0,2}" + re.escape(label) + r"\*{0,2}"
                + r"\s*:\s*\*{0,2}(.+?)\*{0,2}\s*$"
            )

        fields = {
            "org": _pat("Organization"),
            "department": _pat("Department"),
            "doc_id": _pat("Document ID"),
            "version": _pat("Version"),
            "effective_date": _pat("Effective Date"),
        }
        extracted: dict[str, str] = {}
        for key, pattern in fields.items():
            m = re.search(pattern, markdown_text, re.IGNORECASE | re.MULTILINE)
            extracted[key] = m.group(1).strip() if m else "Unknown"

        # Override org with folder name if available
        if org_name and org_name != "Unknown":
            extracted["org"] = org_name

        extracted["source_file"] = source_file
        extracted["ingested_at"] = datetime.now(timezone.utc).isoformat()
        return extracted

    # ── Chunking ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_table_chunk(text: str) -> bool:
        """Detect Markdown pipe tables or raw HTML <table> blocks."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if sum(1 for ln in lines if ln.startswith("|")) >= 2:
            return True
        if re.search(r"<table[\s>]", text, re.IGNORECASE):
            return True
        return False

    @staticmethod
    def _is_empty_section(chunk) -> bool:
        content = re.sub(r"[#\*\-\_`\[\]\(\)]", " ", chunk.page_content)
        content = re.sub(r"\s+", " ", content).strip()
        return len(content.split()) < settings.parent_chunking.empty_section_min_words

    def _split_parents(self, markdown_text: str, doc_meta: dict) -> list:
        """Split into parent chunks by Markdown headers (H1/H2/H3)."""
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=settings.parent_chunking.headers_to_split_on,
        )
        all_parents = splitter.split_text(markdown_text)

        parent_chunks = []
        header_keys = {"Header 1", "Header 2", "Header 3"}

        for chunk in all_parents:
            if self._is_empty_section(chunk):
                parts = [v for k, v in chunk.metadata.items() if k in header_keys]
                name = " > ".join(parts) if parts else "Unknown"
                print(f"  ⚠  Empty section skipped: [{name}]")
            else:
                chunk.metadata.update(doc_meta)
                parent_chunks.append(chunk)

        return parent_chunks

    def _split_children(self, parent_chunks: list) -> list:
        """Split parent chunks into smaller children; keep tables atomic."""
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.child_chunking.chunk_size,
            chunk_overlap=settings.child_chunking.chunk_overlap,
        )
        child_chunks: list = []
        atomic_count = 0

        for chunk in parent_chunks:
            if self._is_table_chunk(chunk.page_content):
                chunk.metadata["chunk_type"] = "table_atomic"
                child_chunks.append(chunk)
                atomic_count += 1
            else:
                chunk.metadata["chunk_type"] = "text"
                child_chunks.extend(child_splitter.split_documents([chunk]))

        print(f"  → {atomic_count} table chunk(s) kept atomic.")
        return child_chunks

    # ── Validation ────────────────────────────────────────────────────────

    @staticmethod
    def _validate_chunks(chunks: list, label: str) -> dict:
        warnings: list[str] = []
        seen: dict[str, int] = {}
        short_count = dupe_count = 0

        for i, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            if len(content) < settings.child_chunking.min_chunk_length:
                short_count += 1
                warnings.append(f"[{label}] Chunk {i+1} too short ({len(content)} chars)")
            h = hashlib.md5(content.encode()).hexdigest()
            if h in seen:
                dupe_count += 1
                warnings.append(f"[{label}] Chunk {i+1} duplicates chunk {seen[h]+1}")
            else:
                seen[h] = i

        return {
            "total": len(chunks),
            "short": short_count,
            "duplicates": dupe_count,
            "warnings": warnings,
        }

    # ── Upload to Qdrant ──────────────────────────────────────────────────

    def _upload_chunks(
        self,
        chunks: list,
        collection_name: str,
        label: str,
    ) -> int:
        """Embed each chunk and upsert into a Qdrant collection."""
        points: list[PointStruct] = []
        print(f"  Embedding {len(chunks)} {label} chunks...")

        for i, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            if not content:
                continue

            # Deterministic UUID — same content always gets the same ID (upsert-safe)
            content_hash = hashlib.md5(content.encode()).hexdigest()
            point_id = str(uuid.UUID(content_hash))

            vector = self._get_embedding(content)
            payload = {"text": content}
            payload.update(chunk.metadata)

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"    {i+1}/{len(chunks)} embedded...")

        if points:
            self.qdrant.upsert(collection_name=collection_name, points=points)
            print(f"  ✓ Uploaded {len(points)} points → '{collection_name}'")

        return len(points)

    def _delete_document_points(self, collection_name: str, org_name: str, source_file: str) -> None:
        """Remove previously indexed chunks for a specific document before re-ingesting it."""
        self.qdrant.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="org", match=MatchValue(value=org_name)),
                        FieldCondition(key="source_file", match=MatchValue(value=source_file)),
                    ]
                )
            ),
            wait=True,
        )

    # ── Per-File Processing ───────────────────────────────────────────────

    def _process_file(self, pdf_path: str, org_name: str) -> dict | None:
        """Full 8-step pipeline for a single PDF file."""
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        normalized_path = str(pdf_path).replace("\\", "/")
        is_sop_document = any(
            marker in normalized_path
            for marker in ("/sop-procedures/", "/faq-docs/")
        )
        parsed_root = settings.PARSED_SOP_DIR if is_sop_document else settings.PARSED_POLICY_DIR
        parsed_folder = "sops" if is_sop_document else "policies"
        output_dir = str(Path(parsed_root) / org_name / "parsed-docs" / parsed_folder)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(pdf_path)}  [org: {org_name}]")
        print(f"{'='*70}")

        # Step 1 — Parse PDF with Docling
        print("\n[Step 1] Parsing PDF with Docling...")
        try:
            raw_markdown = self._parse_pdf(pdf_path)
        except Exception as e:
            print(f"  ERROR parsing {pdf_path}: {e}")
            return None

        raw_path = os.path.join(output_dir, f"{base_name}_raw.md")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_markdown)
        print(f"  Saved raw markdown → {raw_path}")

        # Step 2 — Clean markdown
        print("\n[Step 2] Cleaning markdown (encoding + table repair)...")
        clean_markdown = self._clean_markdown(raw_markdown)

        clean_path = os.path.join(output_dir, f"{base_name}.md")
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(clean_markdown)
        print(f"  Saved clean markdown → {clean_path}")

        # Step 3 — Extract metadata
        print("\n[Step 3] Extracting document metadata...")
        doc_meta = self._extract_metadata(clean_markdown, base_name, org_name)
        print(f"  {json.dumps({k: v for k, v in doc_meta.items() if k != 'ingested_at'})}")

        # Step 4 — Parent splitting
        print("\n[Step 4] Splitting into Parent Chunks (H1/H2/H3)...")
        parent_chunks = self._split_parents(clean_markdown, doc_meta)
        print(f"  Parent chunks: {len(parent_chunks)}")

        # Step 5 — Child splitting
        print("\n[Step 5] Splitting into Child Chunks...")
        child_chunks = self._split_children(parent_chunks)
        print(f"  Child chunks: {len(child_chunks)}")

        # Step 6 — Validation
        print("\n[Step 6] Validating...")
        p_report = self._validate_chunks(parent_chunks, "PARENT")
        c_report = self._validate_chunks(child_chunks, "CHILD")
        all_warnings = p_report["warnings"] + c_report["warnings"]
        if c_report["total"] == p_report["total"] and len(parent_chunks) > 0:
            all_warnings.append(
                "[PIPELINE] Parent == Child count — child splitting may not be working."
            )
        if all_warnings:
            for w in all_warnings:
                print(f"  ⚠  {w}")
        else:
            print("  ✓ All chunks passed quality checks.")

        # Step 7 — Save debug outputs
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
                tag = " [TABLE]" if chunk.metadata.get("chunk_type") == "table_atomic" else ""
                f.write(f"### Child Chunk {i+1}{tag}\n")
                f.write(f"**Metadata**: {chunk.metadata}\n\n")
                f.write(f"```markdown\n{chunk.page_content}\n```\n\n---\n\n")
        print(f"  Saved → {chunks_path}")

        validation = {
            "file": base_name,
            "org": org_name,
            "parent_chunks": p_report,
            "child_chunks": c_report,
        }
        val_path = os.path.join(output_dir, f"{base_name}_validation.json")
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(validation, f, indent=2)
        print(f"  Saved → {val_path}")

        # Step 8 — Upload to Qdrant
        if is_sop_document:
            print("\n[Step 8] SOP Document detected. Uploading to SOP parent collection...")
            self._delete_document_points(settings.qdrant.collection_sop_parents, org_name, base_name)
            n_parents = self._upload_chunks(
                parent_chunks, settings.qdrant.collection_sop_parents, "parent"
            )
        else:
            print("\n[Step 8] Uploading Policy to Qdrant...")
            self._delete_document_points(settings.qdrant.collection_parents, org_name, base_name)
            self._delete_document_points(settings.qdrant.collection_children, org_name, base_name)
            n_parents = self._upload_chunks(
                parent_chunks, settings.qdrant.collection_parents, "parent"
            )
            n_children = self._upload_chunks(
                child_chunks, settings.qdrant.collection_children, "child"
            )

        print(f"\n{'─'*50}")
        print(f"  DONE: {base_name}")
        if is_sop_document:
            print(f"  Parents  : {n_parents} → SOP Parents")
        else:
            print(f"  Parents  : {n_parents} → Policy Parents")
            print(f"  Children : {n_children} → Policy Children")
        print(f"  Warnings : {len(all_warnings)}")
        print(f"{'─'*50}")

        return validation

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, docs_dir: Path | None = None, force: bool = False) -> list[dict]:
        """
        Run the full ingestion pipeline.

        Args:
            docs_dir: Override path to the legacy documents directory. The pipeline
                      also scans POLICY_DOCS_DIR and KNOWLEDGE_DOCS_DIR roots.
                      Expected structure:
                      - root/org1/policy-docs/*.pdf
                      - root/org1/sop-procedures/*.pdf
                      Backward compatible:
                      - root/org1/faq-docs/*.pdf
                      - root/org1/*.pdf
            force:    If True, delete existing collections and re-create them.

        Returns:
            List of per-file validation reports.
        """
        docs_dir = Path(settings.DOCS_DIR) if docs_dir is None else Path(docs_dir)
        candidate_roots: list[Path] = [
            docs_dir,
            Path(settings.POLICY_DOCS_DIR),
            Path(settings.KNOWLEDGE_DOCS_DIR),
        ]

        root_dirs: list[Path] = []
        for root in candidate_roots:
            try:
                resolved = root.resolve()
            except Exception:
                resolved = root
            if resolved in root_dirs:
                continue
            if not resolved.exists() or not resolved.is_dir():
                continue
            root_dirs.append(resolved)

        if force:
            print("\n⚠  Force mode — deleting existing collections...")
            for name in [
                settings.qdrant.collection_parents,
                settings.qdrant.collection_children,
                settings.qdrant.collection_sop_parents,
            ]:
                try:
                    self.qdrant.delete_collection(name)
                    print(f"  Deleted: {name}")
                except Exception:
                    pass
            self._ensure_collections()

        # Discover PDFs from org policy/SOP folders across all configured roots.
        pdf_files: list[tuple[str, str]] = []  # (pdf_path, org_name)
        seen_paths: set[str] = set()

        for root in root_dirs:
            for org_dir in sorted(root.iterdir()):
                if not org_dir.is_dir():
                    continue

                discovered_in_subdirs = False
                for candidate in (
                    org_dir / "policy-docs",
                    org_dir / "sop-procedures",
                    org_dir / "faq-docs",
                ):
                    if not candidate.is_dir():
                        continue

                    discovered_in_subdirs = True
                    for pattern in ("*.pdf", "*.PDF"):
                        for pdf in sorted(candidate.glob(pattern)):
                            key = str(pdf.resolve())
                            if key in seen_paths:
                                continue
                            seen_paths.add(key)
                            pdf_files.append((str(pdf), org_dir.name))

                # Fallback: also check org root for direct PDFs.
                if not discovered_in_subdirs:
                    for pattern in ("*.pdf", "*.PDF"):
                        for pdf in sorted(org_dir.glob(pattern)):
                            key = str(pdf.resolve())
                            if key in seen_paths:
                                continue
                            seen_paths.add(key)
                            pdf_files.append((str(pdf), org_dir.name))

        if not pdf_files:
            for root in root_dirs:
                for pdf in sorted(root.rglob("*")):
                    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
                        continue
                    key = str(pdf.resolve())
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)
                    try:
                        rel = pdf.relative_to(root)
                        org_name = rel.parts[0] if len(rel.parts) > 1 else "Unknown"
                    except Exception:
                        org_name = "Unknown"
                    pdf_files.append((str(pdf), org_name))

        if not pdf_files:
            checked = ", ".join(str(root) for root in root_dirs) or "(no existing roots)"
            print(f"\nNo PDFs found in configured roots: {checked}.")
            return []

        print(f"\nFound {len(pdf_files)} PDF(s). Starting pipeline...\n")

        summaries: list[dict] = []
        for pdf_path, org_name in pdf_files:
            result = self._process_file(pdf_path, org_name)
            if result:
                summaries.append(result)

        # Write pipeline report
        report = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "files_processed": len(summaries),
            "qdrant_url": settings.qdrant.url,
            "embedding_model": settings.embedding.model,
            "embedding_dimension": settings.embedding.dimension,
            "collections": {
                "parents": settings.qdrant.collection_parents,
                "children": settings.qdrant.collection_children,
            },
            "summaries": summaries,
        }
        os.makedirs(str(settings.PARSED_DIR), exist_ok=True)
        report_path = Path(settings.PARSED_DIR) / "_pipeline_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*70}")
        print("Pipeline complete!")
        print(f"  Report       → {report_path}")
        print(f"  Qdrant UI    → {settings.qdrant.url}/dashboard")
        print(f"  Parents      → {settings.qdrant.url}/collections/{settings.qdrant.collection_parents}")
        print(f"  Children     → {settings.qdrant.url}/collections/{settings.qdrant.collection_children}")
        print(f"{'='*70}\n")

        return summaries
