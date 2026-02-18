"""
VocalMind Universal Ingestion Pipeline
============================================
Dual-Granularity RAG Pipeline.

Dependencies:
    pip install llama-cloud>=1.4.0 langchain langchain-text-splitters python-dotenv ftfy
"""

import os
import re
import glob
import json
import hashlib
from datetime import datetime
from dotenv import load_dotenv

try:
    import ftfy
    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False
    print("Warning: 'ftfy' not installed. Encoding fixes will be skipped.")
    print("  Install with: pip install ftfy")

# New unified SDK — replaces the deprecated 'llama-parse' package.
# Install: pip uninstall llama-parse && pip install llama-cloud
from llama_cloud import LlamaCloud
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHILD_CHUNK_SIZE        = 600  # characters — large enough to keep rules intact
CHILD_CHUNK_OVERLAP     = 120  # characters
MIN_CHUNK_LENGTH        = 30   # characters — anything shorter is flagged as junk
EMPTY_SECTION_MIN_WORDS = 4    # word tokens — sections with fewer real words are empty
                                # (e.g. "2.3 Technical Triggers" with no body = 0 words)

# Headers that drive the parent-chunk boundaries
HEADERS_TO_SPLIT_ON = [
    ("#",   "Header 1"),
    ("##",  "Header 2"),
    ("###", "Header 3"),
]

# ---------------------------------------------------------------------------
# Helper: fix encoding artifacts (â€™ → ', &#x26; → &, etc.)
# ---------------------------------------------------------------------------

def fix_encoding(text: str) -> str:
    """
    Apply ftfy + unescape common HTML entities left by parsers.
    Also applies a manual mojibake table for common UTF-8 → Latin-1 → UTF-8
    double-encoding artifacts that appear when ftfy is not available.
    """
    if FTFY_AVAILABLE:
        text = ftfy.fix_text(text)
    else:
        # Manual fallback: common LlamaParse UTF-8 mojibake patterns
        mojibake_map = {
            "â€™": "'",        # RIGHT SINGLE QUOTATION MARK
            "â€œ": "\u201c",   # LEFT DOUBLE QUOTATION MARK
            "â€\x9d": "\u201d", # RIGHT DOUBLE QUOTATION MARK
            "â€\u201c": "\u2013", # EN DASH
            "â€\u201d": "\u2014", # EM DASH
            "â€¦": "\u2026",   # HORIZONTAL ELLIPSIS
            "Â·": "\u00b7",    # MIDDLE DOT
            "Â ": "\u00a0",    # NON-BREAKING SPACE
        }
        for bad, good in mojibake_map.items():
            text = text.replace(bad, good)

    # HTML entities that LlamaParse sometimes leaves behind
    entity_map = {
        "&#x26;": "&",
        "&#x3C;": "<",
        "&#x3E;": ">",
        "&amp;":  "&",
        "&lt;":   "<",
        "&gt;":   ">",
        "&quot;": '"',
        "&#x27;": "'",
    }
    for entity, replacement in entity_map.items():
        text = text.replace(entity, replacement)

    return text


# ---------------------------------------------------------------------------
# Helper: extract document-level metadata from the header section
# ---------------------------------------------------------------------------

def extract_doc_metadata(markdown_text: str, source_file: str) -> dict:
    """
    Parse the preamble block of the document to pull structured fields.

    FIX-B: Patterns now handle BOTH output styles LlamaParse produces:
      - Plain text:  'Organization: NA Telecommunications'
      - Bold Markdown: '**Organization:** NA Telecommunications'

    The optional \\*{0,2} groups match zero, one, or two asterisks on either
    side of the field name and after the colon, making the regex format-agnostic.
    re.MULTILINE ensures ^ and $ anchor correctly to each line.
    """
    def make_pattern(label: str) -> str:
        return (
            r"^\*{0,2}"          # optional opening **
            + re.escape(label)   # literal field name
            + r"\*{0,2}"         # optional closing ** on label
            + r"\s*:\s*"         # colon with optional surrounding spaces
            + r"\*{0,2}"         # optional opening ** on value
            + r"(.+?)"           # captured value (non-greedy)
            + r"\*{0,2}\s*$"     # optional closing ** then end of line
        )

    field_patterns = {
        "org":            make_pattern("Organization"),
        "department":     make_pattern("Department"),
        "doc_id":         make_pattern("Document ID"),
        "version":        make_pattern("Version"),
        "effective_date": make_pattern("Effective Date"),
    }

    extracted = {}
    for key, pattern in field_patterns.items():
        match = re.search(pattern, markdown_text, re.IGNORECASE | re.MULTILINE)
        extracted[key] = match.group(1).strip() if match else "Unknown"

    extracted["source_file"] = source_file
    extracted["ingested_at"] = datetime.utcnow().isoformat() + "Z"

    return extracted


# ---------------------------------------------------------------------------
# Helper: repair orphaned table content pushed outside table blocks by parser
# ---------------------------------------------------------------------------

def repair_orphaned_table_rows(markdown_text: str) -> str:
    """
    LlamaParse occasionally pushes table rows outside the Markdown table fence,
    especially for the last row or rows that span cells. This function detects
    lines that look like table rows (start with '|') that are separated from
    the nearest preceding table block by only blank lines, and re-attaches them.

    Example of broken input:
        | Col A | Col B |
        | ----- | ----- |
        | Val 1 | Val 2 |

        Happy   Match Energy   "That is fantastic!"   <-- orphaned plain text

    This won't fix non-pipe orphans (plain text that was a table cell), but it
    will fix the common case of pipe-rows being separated by blank lines.
    """
    lines  = markdown_text.splitlines()
    output = []
    i      = 0

    while i < len(lines):
        line = lines[i]
        output.append(line)

        # If this line is a table row, look ahead for orphaned rows
        if line.strip().startswith("|"):
            j = i + 1
            # Consume blank lines between table rows
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            # If the next non-blank line is also a table row, bridge the gap
            if j < len(lines) and lines[j].strip().startswith("|"):
                i = j
                continue  # Re-enter loop without incrementing past j

        i += 1

    return "\n".join(output)


# ---------------------------------------------------------------------------
# Helper: detect whether a chunk's content is a table (Markdown OR HTML)
# ---------------------------------------------------------------------------

def is_table_chunk(text: str) -> bool:
    """
    FIX-A: Detects BOTH Markdown pipe tables AND raw HTML <table> blocks.

    LlamaParse inconsistently outputs tables in one of two formats depending
    on the source document's layout complexity:
      - Simple tables  → Markdown pipe syntax  (| Col | Col |)
      - Complex tables → Raw HTML              (<table><tbody><tr><td>)

    Previously only pipe tables were detected, so HTML tables in docs 01 and 02
    were passed to RecursiveCharacterTextSplitter and cut mid-tag, producing
    meaningless chunks like '<tr>\\n    <th>Angry</th>' with no surrounding context.

    Now both formats are detected and kept as atomic (unsplit) chunks.
    """
    # Check 1: Markdown pipe table — at least 2 lines starting with |
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    pipe_table_lines = sum(1 for ln in lines if ln.startswith("|"))
    if pipe_table_lines >= 2:
        return True

    # Check 2: HTML table — contains an opening <table tag (case-insensitive)
    if re.search(r"<table[\s>]", text, re.IGNORECASE):
        return True

    return False


# ---------------------------------------------------------------------------
# Helper: validate a list of chunks, return a report dict
# ---------------------------------------------------------------------------

def validate_chunks(chunks: list, label: str) -> dict:
    """
    Checks:
      - Empty / too-short chunks
      - Duplicate chunks (same content hash)
    Returns a dict with stats and a list of warnings.
    """
    warnings    = []
    seen_hashes = {}
    short_count = 0
    dupe_count  = 0

    for i, chunk in enumerate(chunks):
        content = chunk.page_content.strip()
        length  = len(content)

        # 1. Too short?
        if length < MIN_CHUNK_LENGTH:
            short_count += 1
            warnings.append(
                f"[{label}] Chunk {i+1} is very short ({length} chars): "
                f"'{content[:60]}...'"
            )

        # 2. Duplicate?
        h = hashlib.md5(content.encode()).hexdigest()
        if h in seen_hashes:
            dupe_count += 1
            warnings.append(
                f"[{label}] Chunk {i+1} is a duplicate of chunk "
                f"{seen_hashes[h]+1}: '{content[:60]}'"
            )
        else:
            seen_hashes[h] = i

    return {
        "total":      len(chunks),
        "short":      short_count,
        "duplicates": dupe_count,
        "warnings":   warnings,
    }


# ---------------------------------------------------------------------------
# Helper: detect empty parent sections (section header with no real body)
# ---------------------------------------------------------------------------

def is_empty_section(chunk) -> bool:
    """
    A parent chunk is 'empty' if it contains fewer than EMPTY_SECTION_MIN_WORDS
    real word tokens after stripping markdown syntax and whitespace.
    """
    content = chunk.page_content
    content = re.sub(r"[#\*\-\_`\[\]\(\)]", " ", content)  # strip formatting chars
    content = re.sub(r"\s+", " ", content).strip()
    words   = [w for w in content.split(" ") if w]
    return len(words) < EMPTY_SECTION_MIN_WORDS


# ---------------------------------------------------------------------------
# Core: table-aware child splitting
# ---------------------------------------------------------------------------

def split_into_children(parent_chunks: list, child_splitter) -> list:
    """
    For each parent chunk:
      - If it contains a Markdown OR HTML table → keep it as a single atomic child.
      - Otherwise → apply the RecursiveCharacterTextSplitter normally.
    """
    child_chunks = []
    atomic_count = 0

    for chunk in parent_chunks:
        if is_table_chunk(chunk.page_content):
            chunk.metadata["chunk_type"] = "table_atomic"
            child_chunks.append(chunk)
            atomic_count += 1
        else:
            chunk.metadata["chunk_type"] = "text"
            splits = child_splitter.split_documents([chunk])
            child_chunks.extend(splits)

    print(f"  → {atomic_count} table chunk(s) kept atomic (not split).")
    return child_chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_file(pdf_file: str, parser: LlamaCloud, output_dir: str):
    # parser is an instance of llama_cloud.LlamaCloud
    base_name = os.path.splitext(os.path.basename(pdf_file))[0]
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(pdf_file)}")
    print(f"{'='*80}")

    # ------------------------------------------------------------------
    # STEP 1 — Parse with LlamaCloud
    # ------------------------------------------------------------------
    print("\n[Step 1] Parsing with LlamaCloud...")
    try:
        with open(pdf_file, "rb") as f:
            result = parser.parsing.parse(
                upload_file=f,
                tier="agentic",
                version="latest",
                expand=["markdown", "markdown_full", "text"],
                verbose=True
            )
    except Exception as e:
        print(f"  ERROR: Could not parse {pdf_file}: {e}")
        return

    if not result:
        print("  ERROR: LlamaCloud returned None result.")
        return

    # Extract markdown content — prefer markdown_full, fall back to page list
    if result.markdown_full:
        raw_markdown = result.markdown_full
    elif result.markdown and result.markdown.pages:
        raw_markdown = "\n".join([
            page.markdown for page in result.markdown.pages
            if hasattr(page, "markdown")
        ])
    else:
        print("  ERROR: LlamaCloud returned no content in markdown_full or markdown.pages.")
        print(f"  Result fields set: {result.model_fields_set}")
        return

    # Save raw (pre-fix) markdown for debugging — BEFORE encoding changes
    raw_path = os.path.join(output_dir, f"{base_name}_raw.md")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_markdown)
    print(f"  Saved raw markdown → {raw_path}")

    # ------------------------------------------------------------------
    # STEP 2 — Encoding fix
    # ------------------------------------------------------------------
    print("\n[Step 2] Fixing encoding artifacts...")
    clean_markdown = fix_encoding(raw_markdown)

    # ------------------------------------------------------------------
    # STEP 2b — Repair orphaned table rows
    # ------------------------------------------------------------------
    print("\n[Step 2b] Repairing orphaned table rows...")
    clean_markdown = repair_orphaned_table_rows(clean_markdown)

    # Save the cleaned version
    clean_path = os.path.join(output_dir, f"{base_name}.md")
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(clean_markdown)
    print(f"  Saved clean markdown → {clean_path}")

    # ------------------------------------------------------------------
    # STEP 3 — Extract document-level metadata
    # ------------------------------------------------------------------
    print("\n[Step 3] Extracting document metadata...")
    doc_meta = extract_doc_metadata(clean_markdown, base_name)
    print(f"  Detected metadata: {json.dumps(doc_meta, indent=4)}")

    # ------------------------------------------------------------------
    # STEP 4 — Parent splitting (header-based)
    # ------------------------------------------------------------------
    print("\n[Step 4] Splitting by Markdown headers (Parent Chunks)...")
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON
    )
    all_parent_chunks = markdown_splitter.split_text(clean_markdown)
    print(f"  Total sections parsed: {len(all_parent_chunks)}")

    # Detect and flag empty sections
    empty_sections = []
    parent_chunks  = []
    HEADER_KEYS    = {"Header 1", "Header 2", "Header 3"}

    for chunk in all_parent_chunks:
        if is_empty_section(chunk):
            header_parts = [v for k, v in chunk.metadata.items() if k in HEADER_KEYS]
            section_name = " > ".join(header_parts) if header_parts else "Unknown Section"
            empty_sections.append(section_name)
            print(f"  ⚠  EMPTY SECTION detected and skipped: [{section_name}]")
        else:
            chunk.metadata.update(doc_meta)
            parent_chunks.append(chunk)

    print(f"  Parent chunks after filtering empties: {len(parent_chunks)}")

    # ------------------------------------------------------------------
    # STEP 5 — Child splitting (character-based, table-aware)
    # ------------------------------------------------------------------
    print("\n[Step 5] Splitting into Child Chunks (table-aware)...")
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )
    child_chunks = split_into_children(parent_chunks, child_splitter)
    print(f"  Total child chunks: {len(child_chunks)}")

    # ------------------------------------------------------------------
    # STEP 6 — Validation
    # ------------------------------------------------------------------
    print("\n[Step 6] Validating chunk quality...")
    parent_report = validate_chunks(parent_chunks, "PARENT")
    child_report  = validate_chunks(child_chunks,  "CHILD")

    all_warnings = parent_report["warnings"] + child_report["warnings"]

    # Warn if parent == child count — likely means child splitting isn't running
    if len(parent_chunks) > 0:
        if child_report["total"] == parent_report["total"]:
            eq_warning = (
                f"[PIPELINE] Parent count ({parent_report['total']}) == Child count "
                f"({child_report['total']}). Child splitting may not be running — "
                f"verify chunk_size={CHILD_CHUNK_SIZE} vs actual content length."
            )
            all_warnings.append(eq_warning)
            print(f"  ⚠  {eq_warning}")

    if all_warnings:
        print(f"  ⚠  {len(all_warnings)} quality warning(s) found:")
        for w in all_warnings:
            print(f"    - {w}")
    else:
        print("  ✓ All chunks passed quality checks.")

    validation_summary = {
        "file":           base_name,
        "empty_sections": empty_sections,
        "parent_chunks":  parent_report,
        "child_chunks":   child_report,
    }

    # ------------------------------------------------------------------
    # STEP 7 — Save outputs
    # ------------------------------------------------------------------
    print("\n[Step 7] Saving outputs...")

    # Chunks analysis markdown (human-readable)
    chunks_path = os.path.join(output_dir, f"{base_name}_chunks.md")
    with open(chunks_path, "w", encoding="utf-8") as f:
        f.write(f"# Chunk Analysis: {base_name}\n\n")
        f.write(f"**Ingested at**: {doc_meta['ingested_at']}  \n")
        f.write(f"**Organization**: {doc_meta['org']}  \n")
        f.write(f"**Document ID**: {doc_meta['doc_id']}  \n")
        f.write(f"**Effective Date**: {doc_meta['effective_date']}  \n\n")

        if empty_sections:
            f.write("## ⚠ Empty Sections (Skipped)\n")
            for s in empty_sections:
                f.write(f"- `{s}`\n")
            f.write("\n")

        f.write(f"## Parent Chunks ({len(parent_chunks)})\n\n")
        for i, chunk in enumerate(parent_chunks):
            f.write(f"### Parent Chunk {i+1}\n")
            f.write(f"**Metadata**: {chunk.metadata}\n\n")
            f.write(f"```markdown\n{chunk.page_content}\n```\n\n---\n\n")

        f.write(f"## Child Chunks ({len(child_chunks)})\n\n")
        for i, chunk in enumerate(child_chunks):
            chunk_type = chunk.metadata.get("chunk_type", "text")
            label = " 🔒 [TABLE — ATOMIC]" if chunk_type == "table_atomic" else ""
            f.write(f"### Child Chunk {i+1}{label}\n")
            f.write(f"**Metadata**: {chunk.metadata}\n\n")
            f.write(f"```markdown\n{chunk.page_content}\n```\n\n---\n\n")

    print(f"  Saved chunks analysis → {chunks_path}")

    # Validation report (JSON — machine-readable)
    validation_path = os.path.join(output_dir, f"{base_name}_validation.json")
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(validation_summary, f, indent=2)
    print(f"  Saved validation report → {validation_path}")

    # Summary to console
    print(f"\n{'─'*60}")
    print(f"  SUMMARY for {base_name}")
    print(f"{'─'*60}")
    print(f"  Empty sections flagged : {len(empty_sections)}")
    print(f"  Parent chunks          : {len(parent_chunks)}")
    print(f"  Child chunks           : {len(child_chunks)}")
    print(f"  Quality warnings       : {len(all_warnings)}")
    print(f"{'─'*60}\n")

    return validation_summary


def main():
    load_dotenv()

    # Locate input docs
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NA_docs")
    if not os.path.exists(docs_dir):
        print(f"Error: '{docs_dir}' not found.")
        print("Please place your PDF files inside a folder named 'NA_docs' "
              "in the same directory as this script.")
        return

    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        print("Warning: LLAMA_CLOUD_API_KEY not found. LlamaParse may fail.")

    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parsed_docs")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize parser — uses llama-cloud (replaces deprecated llama-cloud-services)
    print("Initializing LlamaCloud...")
    try:
        parser = LlamaCloud(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        )
    except Exception as e:
        print(f"Error initializing LlamaCloud: {e}")
        return

    # Find PDFs
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{docs_dir}'.")
        return

    print(f"Found {len(pdf_files)} PDF file(s). Starting pipeline...\n")

    # Process each file
    all_summaries = []
    for pdf_file in pdf_files:
        summary = process_file(pdf_file, parser, output_dir)
        if summary:
            all_summaries.append(summary)

    # Pipeline-wide report
    pipeline_report_path = os.path.join(output_dir, "_pipeline_report.json")
    with open(pipeline_report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_at":    datetime.utcnow().isoformat() + "Z",
                "files":     len(pdf_files),
                "summaries": all_summaries,
            },
            f,
            indent=2,
        )
    print(f"\nPipeline complete. Full report saved → {pipeline_report_path}")


if __name__ == "__main__":
    main()