import json
from rag_app.ingestion.herb import HERBDataLoader


def test_herb_loader_metadata_resolution(
    mock_data_dir, employee_meta, sample_json_data
):
    # Setup metadata files
    meta_dir = mock_data_dir / "metadata"
    meta_dir.mkdir()
    with open(meta_dir / "employee.json", "w") as f:
        json.dump(employee_meta, f)

    # Setup product file
    prod_dir = mock_data_dir / "products"
    prod_dir.mkdir()
    prod_file = prod_dir / "test_prod.json"
    with open(prod_file, "w") as f:
        json.dump(sample_json_data, f)

    loader = HERBDataLoader(mock_data_dir)
    docs = loader.load_all()

    assert len(docs) >= 2
    # Check if ID was resolved
    slack_doc = next(d for d in docs if d.metadata["type"] == "slack")
    assert "Alice Smith" in slack_doc.text
    assert "eid_123" in slack_doc.text
