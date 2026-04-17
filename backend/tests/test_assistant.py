"""
Unit tests for the Manager Assistant endpoint.

Test strategy:
1. Schema validation tests (422 checks) — no DB required
2. Unit tests for IntentResolver — mock Gemini directly, no HTTP or DB needed
"""
import pytest
from unittest.mock import AsyncMock, patch
from uuid import UUID

from app.api.deps import get_current_user
from app.models.enums import UserRole
from app.models.user import User


@pytest.fixture(autouse=True)
def _assistant_auth_manager(client):
    """Assistant routes require a signed-in manager."""
    mgr_id = UUID("b0000000-0000-0000-0000-000000000001")
    org_id = UUID("a0000000-0000-0000-0000-000000000001")

    async def _override():
        return User(
            id=mgr_id,
            organization_id=org_id,
            email="manager@test.local",
            password_hash="x",
            name="Test Manager",
            role=UserRole.manager,
            is_active=True,
        )

    client.app.dependency_overrides[get_current_user] = _override
    yield
    client.app.dependency_overrides.pop(get_current_user, None)


def _post_query(client, query_text: str, mode: str = "chat"):
    return client.post(
        "/api/v1/assistant/query",
        json={"query_text": query_text, "mode": mode},
    )


# ---------------------------------------------------------------------------
# Schema validation tests (no DB needed)
# ---------------------------------------------------------------------------

def test_assistant_query_missing_query_text_returns_422(client):
    """Missing required query_text should return 422 Unprocessable Entity."""
    response = client.post("/api/v1/assistant/query", json={"mode": "chat"})
    assert response.status_code == 422


def test_assistant_query_invalid_mode_returns_422(client):
    """Unknown mode enum value should return 422."""
    response = _post_query(client, "Hello", mode="unknown_mode")
    assert response.status_code == 422


def test_parse_sql_extracts_from_markdown_fence():
    from app.api.routes.assistant import _parse_sql_from_model_output

    raw = "Here you go:\n```sql\nSELECT 1 AS one;\n```\n"
    assert _parse_sql_from_model_output(raw) == "SELECT 1 AS one"


def test_parse_sql_accepts_with_cte():
    from app.api.routes.assistant import _parse_sql_from_model_output

    sql = "WITH t AS (SELECT 1 AS n) SELECT n FROM t LIMIT 1"
    assert _parse_sql_from_model_output(sql) == sql


def test_parse_sql_strips_thinking_tags():
    from app.api.routes.assistant import _parse_sql_from_model_output

    raw = "<thinking>plan</thinking>\nSELECT 2 AS two"
    assert _parse_sql_from_model_output(raw) == "SELECT 2 AS two"


def test_ordinal_followup_offset_parses_second():
    from app.api.routes.assistant import _ordinal_followup_offset

    assert _ordinal_followup_offset("Who is the second one") == 1
    assert _ordinal_followup_offset("show 3rd result") == 2
    assert _ordinal_followup_offset("first one") == 0
    assert _ordinal_followup_offset("top agents") is None


def test_build_ordinal_followup_sql_removes_limit_and_offsets():
    from app.api.routes.assistant import _build_ordinal_followup_sql

    prev = "SELECT name, score FROM leaderboard ORDER BY score DESC LIMIT 1"
    sql = _build_ordinal_followup_sql(prev, 1)
    assert sql is not None
    assert "OFFSET 1" in sql
    assert "LIMIT 1" in sql
    assert "WITH prev AS (SELECT name, score FROM leaderboard ORDER BY score DESC)" in sql


def test_deterministic_rank_answer_is_non_hallucinated():
    from app.api.routes.assistant import _deterministic_rank_answer

    answer = _deterministic_rank_answer([{"name": "Sara Agent", "avg_score": 70.0}])
    assert "Sara Agent" in answer
    assert "70.0" in answer
    assert "only agent" not in answer.lower()


# ---------------------------------------------------------------------------
# Unit tests for IntentResolver (no HTTP, no DB)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_sql_returns_none_for_unknown():
    """resolve_sql should return None when Gemini outputs UNKNOWN."""
    from app.api.routes.assistant import IntentResolver

    resolver = IntentResolver()

    with patch.object(
        resolver,
        "_generate_content_with_fallback",
        new_callable=AsyncMock,
        return_value="UNKNOWN",
    ):
        result = await resolver.resolve_sql("What is the weather?", UUID("00000000-0000-0000-0000-000000000001"))
        assert result is None


@pytest.mark.asyncio
async def test_resolve_sql_strips_markdown_fences():
    """resolve_sql should strip ```sql fences if the LLM adds them."""
    from app.api.routes.assistant import IntentResolver

    resolver = IntentResolver()

    with patch.object(
        resolver,
        "_generate_content_with_fallback",
        new_callable=AsyncMock,
        return_value="```sql\nSELECT 1\n```",
    ):
        result = await resolver.resolve_sql("Simple query", UUID("00000000-0000-0000-0000-000000000001"))
        assert result == "SELECT 1"


@pytest.mark.asyncio
async def test_resolve_sql_rejects_unsafe_statements():
    """resolve_sql must return None for any non-SELECT statement."""
    from app.api.routes.assistant import IntentResolver

    resolver = IntentResolver()

    for unsafe_sql in [
        "DROP TABLE users",
        "DELETE FROM interactions",
        "UPDATE users SET role='admin'",
        "INSERT INTO users VALUES (1, 'hacker')",
    ]:
        with patch.object(
            resolver,
            "_generate_content_with_fallback",
            new_callable=AsyncMock,
            return_value=unsafe_sql,
        ):
            result = await resolver.resolve_sql("bad intent", UUID("00000000-0000-0000-0000-000000000001"))
            assert result is None, f"Should have rejected: {unsafe_sql}"


@pytest.mark.asyncio
async def test_resolve_sql_returns_valid_select():
    """resolve_sql should pass through a valid SELECT statement."""
    from app.api.routes.assistant import IntentResolver

    resolver = IntentResolver()
    valid_sql = "SELECT name FROM users WHERE role = 'agent' LIMIT 5"

    with patch.object(
        resolver,
        "_generate_content_with_fallback",
        new_callable=AsyncMock,
        return_value=valid_sql,
    ):
        result = await resolver.resolve_sql("List agents", UUID("00000000-0000-0000-0000-000000000001"))
        assert result == valid_sql


@pytest.mark.asyncio
async def test_synthesize_answer_returns_string():
    """synthesize_answer should return the LLM text as-is."""
    from app.api.routes.assistant import IntentResolver

    resolver = IntentResolver()
    expected_answer = "There are 3 top agents: Alice, Bob, and Carol."

    with patch.object(
        resolver,
        "_generate_content_with_fallback",
        new_callable=AsyncMock,
        return_value=expected_answer,
    ):
        result = await resolver.synthesize_answer(
            "Who are the top agents?",
            "SELECT name FROM users ...",
            [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}],
        )
        assert result == expected_answer


@pytest.mark.asyncio
async def test_synthesize_answer_fallback_on_exception():
    """synthesize_answer should return a fallback string when Gemini call fails."""
    from app.api.routes.assistant import IntentResolver

    resolver = IntentResolver()

    with patch.object(
        resolver,
        "_generate_content_with_fallback",
        new_callable=AsyncMock,
        side_effect=Exception("API error"),
    ):
        result = await resolver.synthesize_answer(
            "Who are agents?", "SELECT ...", [{"name": "Alice"}]
        )
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.asyncio
async def test_synthesize_answer_fallback_on_empty_rows():
    """synthesize_answer with empty rows should not crash and returns something useful."""
    from app.api.routes.assistant import IntentResolver

    resolver = IntentResolver()

    with patch.object(
        resolver,
        "_generate_content_with_fallback",
        new_callable=AsyncMock,
        side_effect=Exception("API error"),
    ):
        result = await resolver.synthesize_answer("Show me data", "SELECT ...", [])
        assert isinstance(result, str)
        assert len(result) > 0
