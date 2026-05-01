from unittest.mock import patch
from types import SimpleNamespace

from app.llm_trigger import retrieval


def test_read_manual_org_sop_docs_uses_docling_converted_markdown_for_pdf(
    monkeypatch, tmp_path
):
    docs_root = tmp_path / "sop-standards"
    sop_dir = docs_root / "nexalink" / "sop-procedures"
    sop_dir.mkdir(parents=True)

    parsed_dir = docs_root / "nexalink" / "parsed-docs"
    parsed_dir.mkdir(parents=True)

    pdf_path = sop_dir / "01-refund-request-processing.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    (parsed_dir / "01-refund-request-processing.md").write_text(
        "# Refund Request Processing\n\n1. Verify account.\n2. Validate eligibility.",
        encoding="utf-8",
    )

    monkeypatch.setattr(retrieval.settings, "SOP_DOCS_ROOT", str(docs_root))
    monkeypatch.setattr(retrieval.settings, "SOP_PARSED_DOCS_ROOT", str(docs_root))

    text = retrieval._read_manual_org_sop_docs("nexalink")

    assert "[01-refund-request-processing.pdf]" in text
    assert "Refund Request Processing" in text
    assert "Validate eligibility" in text


def test_resolve_retrieved_sop_context_preserves_manual_chunk_metadata(monkeypatch, tmp_path):
    docs_root = tmp_path / "sop-standards"
    sop_dir = docs_root / "nexalink" / "sop-procedures"
    sop_dir.mkdir(parents=True)

    parsed_dir = docs_root / "nexalink" / "parsed-docs"
    parsed_dir.mkdir(parents=True)

    (sop_dir / "01-refund.pdf").write_bytes(b"%PDF-1.4")
    (parsed_dir / "01-refund.md").write_text("# SOP from parsed docs", encoding="utf-8")

    monkeypatch.setattr(retrieval.settings, "SOP_DOCS_ROOT", str(docs_root))
    monkeypatch.setattr(retrieval.settings, "SOP_PARSED_DOCS_ROOT", str(docs_root))

    context = retrieval.resolve_retrieved_sop_context(
        transcript_text="customer asked for refund",
        retrieved_sop_from_pinecone=None,
        org_filter="nexalink",
    )

    assert context.source == "manual"
    assert context.chunks
    assert context.chunks[0].metadata["source_file"] == "01-refund.pdf"
    assert context.chunks[0].metadata["doc_type"] == "sop"
    assert context.chunks[0].metadata["policy_ref"] == []
    assert "SOP from parsed docs" in context.text


def test_read_manual_org_sop_docs_falls_back_to_direct_text_files(monkeypatch, tmp_path):
    docs_root = tmp_path / "sop-standards"
    sop_dir = docs_root / "nexalink" / "sop-procedures"
    sop_dir.mkdir(parents=True)

    (sop_dir / "legacy-sop.md").write_text(
        "# Legacy SOP\n\n1. Step one\n2. Step two", encoding="utf-8"
    )

    monkeypatch.setattr(retrieval.settings, "SOP_DOCS_ROOT", str(docs_root))
    monkeypatch.setattr(retrieval.settings, "SOP_PARSED_DOCS_ROOT", str(docs_root))

    text = retrieval._read_manual_org_sop_docs("nexalink")

    assert "[legacy-sop.md]" in text
    assert "Legacy SOP" in text


def test_resolve_retrieved_sop_prefers_manual_docs_before_qdrant(monkeypatch, tmp_path):
    docs_root = tmp_path / "sop-standards"
    sop_dir = docs_root / "nexalink" / "sop-procedures"
    sop_dir.mkdir(parents=True)

    parsed_dir = docs_root / "nexalink" / "parsed-docs"
    parsed_dir.mkdir(parents=True)

    (sop_dir / "01-refund.pdf").write_bytes(b"%PDF-1.4")
    (parsed_dir / "01-refund.md").write_text("# SOP from parsed docs", encoding="utf-8")

    monkeypatch.setattr(retrieval.settings, "SOP_DOCS_ROOT", str(docs_root))
    monkeypatch.setattr(retrieval.settings, "SOP_PARSED_DOCS_ROOT", str(docs_root))

    with patch.object(retrieval.SOPRetriever, "retrieve_sop", return_value="from-qdrant") as mock_retrieve:
        resolved = retrieval.resolve_retrieved_sop(
            transcript_text="customer asked for refund",
            retrieved_sop_from_pinecone=None,
            org_filter="nexalink",
        )

    assert "SOP from parsed docs" in resolved
    mock_retrieve.assert_not_called()


def test_resolve_retrieved_sop_context_selects_best_matching_manual_doc(monkeypatch, tmp_path):
    docs_root = tmp_path / "sop-standards"
    sop_dir = docs_root / "nexalink" / "sop-procedures"
    sop_dir.mkdir(parents=True)

    parsed_dir = docs_root / "nexalink" / "parsed-docs"
    parsed_dir.mkdir(parents=True)

    (sop_dir / "01-refund-request-processing.pdf").write_bytes(b"%PDF-1.4")
    (parsed_dir / "01-refund-request-processing.md").write_text(
        "# Refund Request\n\n1. Verify refund eligibility.\n2. Confirm refund timeline.\n3. Close with next steps.",
        encoding="utf-8",
    )

    (sop_dir / "02-billing-issue-resolution.pdf").write_bytes(b"%PDF-1.4")
    (parsed_dir / "02-billing-issue-resolution.md").write_text(
        "# Billing Issue\n\n1. Explain charge source.\n2. Confirm customer understanding.\n3. Close with follow-up path.",
        encoding="utf-8",
    )

    monkeypatch.setattr(retrieval.settings, "SOP_DOCS_ROOT", str(docs_root))
    monkeypatch.setattr(retrieval.settings, "SOP_PARSED_DOCS_ROOT", str(docs_root))

    context = retrieval.resolve_retrieved_sop_context(
        transcript_text="customer asked for a refund and the agent explained the refund timeline and goodwill credit",
        retrieved_sop_from_pinecone=None,
        org_filter="nexalink",
    )

    assert "01-refund-request-processing.pdf" in context.text
    assert "02-billing-issue-resolution.pdf" not in context.text
    assert context.chunks[0].metadata["source_file"] == "01-refund-request-processing.pdf"


def test_sop_retriever_excludes_chunks_without_doc_type(monkeypatch, caplog):
    calls = []

    class _FakeClient:
        def query_points(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        score=0.74,
                        payload={
                            "text": "Legacy SOP step without doc type",
                            "source_file": "legacy.md",
                        },
                    )
                ]
            )

    monkeypatch.setattr(retrieval.QdrantRetriever, "_embed_query", lambda self, text: [0.1, 0.2])
    retriever = retrieval.SOPRetriever(client=_FakeClient())

    chunks = retriever.retrieve_sop_chunks("refund request", org_filter="nexalink")

    assert len(chunks) == 0
    assert calls[0]["query_filter"].must[1].key == "doc_type"
    assert calls[0]["query_filter"].must[1].match.value == "sop"
    assert len(calls) == 1
    assert "Chunk missing doc_type metadata \u2014 excluded from results." in caplog.text


def test_policy_retriever_filters_by_policy_doc_type(monkeypatch):
    calls = []

    class _FakeClient:
        def query_points(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        score=0.91,
                        payload={
                            "text": "Policy rule",
                            "doc_type": "policy",
                            "rule_id": "CS-RULE-003",
                        },
                    )
                ]
            )

    monkeypatch.setattr(retrieval.QdrantRetriever, "_embed_query", lambda self, text: [0.1, 0.2])
    retriever = retrieval.PolicyRetriever(client=_FakeClient())

    chunks = retriever.retrieve_policy_chunks("refund claim", org_filter="nexalink")

    assert chunks[0].metadata["doc_type"] == "policy"
    assert calls[0]["query_filter"].must[1].key == "doc_type"
    assert calls[0]["query_filter"].must[1].match.value == "policy"
