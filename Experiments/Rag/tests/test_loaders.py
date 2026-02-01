import json
from rag_app.ingestion.loaders import CustomDirectoryLoader


def test_sidecar_metadata_merging(tmp_path):
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Context body")

    # Create sidecar meta
    meta_file = tmp_path / "test.txt.meta.json"
    meta_payload = {"author": "Tester", "category": "Unit Test"}
    meta_file.write_text(json.dumps(meta_payload))

    loader = CustomDirectoryLoader(input_dir=str(tmp_path))
    docs = loader.load_data()

    assert len(docs) == 1
    assert docs[0].metadata["author"] == "Tester"
    assert docs[0].metadata["category"] == "Unit Test"
    # Verify heuristic title
    assert docs[0].metadata["title"] == "Test"


def test_heuristic_title_generation(tmp_path):
    test_file = tmp_path / "my_cool_document.txt"
    test_file.write_text("content")

    loader = CustomDirectoryLoader(input_dir=str(tmp_path))
    docs = loader.load_data()

    assert docs[0].metadata["title"] == "My Cool Document"
