import pytest


@pytest.fixture
def mock_data_dir(tmp_path):
    """Create a temporary data directory."""
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def sample_json_data():
    """Generic nested JSON for testing extraction."""
    return {
        "slack": [
            {
                "id": "msg_1",
                "Message": {"User": {"userId": "eid_123", "text": "Hello world"}},
                "ThreadReplies": [],
            }
        ],
        "documents": [
            {
                "id": "doc_1",
                "title": "Technical Spec",
                "content": "This is a test document.",
            }
        ],
    }


@pytest.fixture
def employee_meta():
    """Sample metadata for ID resolution."""
    return [
        {"id": "eid_123", "name": "Alice Smith"},
        {"id": "eid_456", "name": "Bob Jones"},
    ]
