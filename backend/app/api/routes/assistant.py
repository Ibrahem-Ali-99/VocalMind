import time
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from sqlmodel import text
from uuid import UUID

logger = logging.getLogger(__name__)

from app.core.config import settings
from google import genai
from google.genai import types

from sqlalchemy.ext.asyncio import AsyncSession as SAAsyncSession
from app.core.database import engine
from app.api.deps import SessionDep
from app.models.enums import QueryMode
from app.schemas.assistant import AssistantQueryRequest

router = APIRouter()

# ---------------------------------------------------------------------------
# Full schema definition
# ---------------------------------------------------------------------------
_SCHEMA_DEFINITION = """
### VocalMind Call Center Database Schema

**Table: organizations** — id UUID PK, name VARCHAR

**Table: users**
- id UUID PK, name VARCHAR, email VARCHAR
- role VARCHAR ('manager' | 'agent')
- organization_id UUID FK → organizations.id

**Table: interactions**
- id UUID PK, organization_id UUID FK, agent_id UUID FK → users.id
- duration_seconds INTEGER, interaction_date TIMESTAMP
- processing_status VARCHAR ('completed' | 'pending' | 'failed')
- language_detected VARCHAR, has_overlap BOOLEAN

**Table: interaction_scores**
- id UUID PK, interaction_id UUID FK → interactions.id
- overall_score FLOAT (0–1), empathy_score FLOAT (0–1)
- policy_score FLOAT (0–1), resolution_score FLOAT (0–1)
- was_resolved BOOLEAN

**Table: company_policies** — id UUID PK, policy_category VARCHAR, policy_title VARCHAR, policy_text TEXT, is_active BOOLEAN

**Table: organization_policies** — id UUID PK, organization_id UUID FK, policy_id UUID FK, is_active BOOLEAN

**Table: policy_compliance**
- id UUID PK, interaction_id UUID FK, policy_id UUID FK
- is_compliant BOOLEAN, compliance_score FLOAT (0–1), llm_reasoning TEXT

**Table: utterances**
- id UUID PK, interaction_id UUID FK
- speaker VARCHAR ('agent' | 'customer')
- emotion VARCHAR ('neutral' | 'happy' | 'frustrated' | 'angry' | 'sad' | 'empathetic' | 'fearful')
- start_time FLOAT, end_time FLOAT
"""


def _build_sql_system_prompt(org_id: UUID) -> str:
    return f"""You are an expert PostgreSQL data analyst for VocalMind, a call-center AI monitoring platform.
Given a natural language question, write a single optimized PostgreSQL SELECT query to answer it.

{_SCHEMA_DEFINITION}

### Critical Rules:
1. ALWAYS filter organization scope: use `organization_id = '{str(org_id)}'` on users and interactions.
2. Output ONLY the raw SQL — no markdown fences, no explanations.
3. If the question cannot be answered, output exactly: UNKNOWN
4. SELECT only — no INSERT, UPDATE, DELETE, DROP, TRUNCATE, DDL.
5. Add LIMIT 50 unless user asks for more or query uses aggregates (COUNT/AVG/SUM).
6. Scores are 0.0–1.0 floats. Multiply by 10 for a 0–10 scale.
7. Use ILIKE for case-insensitive name matching.
8. Cast UUIDs to TEXT (::text) in output.

### Examples:
Q: Who are the top 5 agents by overall score this week?
A: SELECT u.name, ROUND(AVG(s.overall_score) * 10, 1) AS avg_score FROM users u JOIN interactions i ON i.agent_id = u.id AND i.organization_id = '{str(org_id)}' JOIN interaction_scores s ON s.interaction_id = i.id WHERE u.role = 'agent' AND i.interaction_date >= date_trunc('week', now()) GROUP BY u.id, u.name ORDER BY avg_score DESC LIMIT 5

Q: How many policy violations this week?
A: SELECT COUNT(*) AS violation_count FROM policy_compliance pc JOIN interactions i ON pc.interaction_id = i.id WHERE pc.is_compliant = false AND i.organization_id = '{str(org_id)}' AND i.interaction_date >= date_trunc('week', now())

Q: Most common customer emotions?
A: SELECT emotion, COUNT(*) AS count FROM utterances u JOIN interactions i ON u.interaction_id = i.id WHERE u.speaker = 'customer' AND i.organization_id = '{str(org_id)}' GROUP BY emotion ORDER BY count DESC LIMIT 10
"""


def _build_synthesis_prompt(user_query: str, sql: str, rows: list) -> str:
    return f"""You are a helpful business analyst assistant for a call-center manager.
The manager asked: "{user_query}"

SQL executed:
{sql}

Results ({len(rows)} row(s), up to 20 shown):
{rows[:20]}

Write a concise natural language answer (2–4 sentences) directly addressing the question using the data.
- Be specific: mention names, numbers, percentages.
- If empty, explain why (no data yet, filter too strict, etc.).
- Plain paragraph text only — no markdown, no bullet points, no SQL repetition.
"""


class IntentResolver:
    """LLM-based Text-to-SQL + answer synthesis using Gemini."""

    def __init__(self):
        self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    async def resolve_sql(self, query: str, org_id: UUID) -> Optional[str]:
        """Convert natural language to a safe PostgreSQL SELECT."""
        prompt = _build_sql_system_prompt(org_id) + f"\n\nUser Question: {query}\nSQL:"
        try:
            response = await self._client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            sql = (response.text or "").strip()

            for fence in ("```sql", "```"):
                if sql.startswith(fence):
                    sql = sql[len(fence):]
            if sql.endswith("```"):
                sql = sql[:-3]
            sql = sql.strip()

            if sql.upper() == "UNKNOWN":
                return None

            safe = sql.lower()
            forbidden = ("drop ", "delete ", "update ", "insert ", "truncate ", "alter ", "create ")
            if not safe.startswith("select") or any(kw in safe for kw in forbidden):
                logger.error(f"Unsafe SQL generated: {sql}")
                return None

            return sql
        except Exception as exc:
            logger.error(f"SQL generation via Gemini failed: {exc}")
            return None

    async def synthesize_answer(self, user_query: str, sql: str, rows: list) -> str:
        """Produce a human-readable answer from raw SQL results."""
        prompt = _build_synthesis_prompt(user_query, sql, rows)
        try:
            response = await self._client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3),
            )
            return (response.text or "").strip()
        except Exception as exc:
            logger.warning(f"Answer synthesis failed: {exc}")
            return f"I found {len(rows)} result(s)." if rows else "The query returned no results."


@router.post("/query")
async def process_assistant_query(
    request: AssistantQueryRequest,
    session: SessionDep,  # kept for FastAPI DI; actual DB uses raw SA engine
):
    """Process natural language queries from the Manager Assistant."""
    query_text = request.query_text
    mode = request.mode or QueryMode.chat

    # Use a raw connection to avoid ALL SQLModel/ORM lazy-loading issues
    async with engine.connect() as conn:

        # 1. Find the manager (raw SQL — no ORM, no lazy loading)
        r = await conn.execute(
            text("SELECT id, organization_id FROM users WHERE role = 'manager' LIMIT 1")
        )
        row = r.first()
        if not row:
            r = await conn.execute(text("SELECT id, organization_id FROM users LIMIT 1"))
            row = r.first()

        if not row:
            raise HTTPException(status_code=404, detail="No user found to attribute query")

        manager_id = row[0]
        org_id = row[1]
        start_time = time.time()
        resolver = IntentResolver()

        # 2. Recent conversational context (last 5 min)
        ctx_r = await conn.execute(
            text("""
                SELECT query_text, created_at FROM assistant_queries
                WHERE user_id = :uid AND ai_understanding IS NOT NULL
                ORDER BY created_at DESC LIMIT 1
            """),
            {"uid": str(manager_id)},
        )
        ctx_row = ctx_r.first()
        context_prefix = ""
        if ctx_row and (time.time() - ctx_row[1].timestamp() < 300):
            context_prefix = (
                f"Conversational context — the user's previous question was: "
                f"'{ctx_row[0]}'. Use this only if the current question "
                f"references it with pronouns or relative terms. "
            )

        extended_query = context_prefix + query_text

        # 3. Generate SQL via Gemini
        sql = await resolver.resolve_sql(extended_query, org_id)

        if not sql:
            response_text = (
                "I'm sorry, I can't answer that with the available data. "
                "Try asking about agents, interactions, scores, compliance, or emotions."
            )
            await conn.execute(
                text("""
                    INSERT INTO assistant_queries
                    (user_id, organization_id, query_mode, query_text, response_text, execution_time_ms)
                    VALUES (:uid, :oid, :mode, :qt, :rt, :etms)
                """),
                {
                    "uid": str(manager_id), "oid": str(org_id), "mode": mode.value,
                    "qt": query_text, "rt": response_text,
                    "etms": int((time.time() - start_time) * 1000),
                },
            )
            await conn.commit()
            return {"type": "ai", "content": response_text, "mode": mode.value, "success": False}

        # 4. Execute the generated SQL
        try:
            exec_start = time.time()
            result = await conn.execute(text(sql))
            rows = [dict(r._mapping) for r in result]
            execution_time_ms = int((time.time() - exec_start) * 1000)
        except Exception as exc:
            await conn.rollback()
            logger.error(f"SQL exec error — {sql!r} — {exc}")
            err_response = (
                "I understood your request but hit a database error. "
                "Try rephrasing or simplifying the question."
            )
            await conn.execute(
                text("""
                    INSERT INTO assistant_queries
                    (user_id, organization_id, query_mode, query_text, response_text, execution_time_ms)
                    VALUES (:uid, :oid, :mode, :qt, :rt, :etms)
                """),
                {
                    "uid": str(manager_id), "oid": str(org_id), "mode": mode.value,
                    "qt": query_text, "rt": err_response,
                    "etms": int((time.time() - start_time) * 1000),
                },
            )
            await conn.commit()
            return {"type": "ai", "content": err_response, "mode": mode.value, "sql": sql, "success": False}

        # 5. Synthesize natural language answer
        natural_answer = await resolver.synthesize_answer(query_text, sql, rows)

        ins_r = await conn.execute(
            text("""
                INSERT INTO assistant_queries
                (user_id, organization_id, query_mode, query_text,
                 ai_understanding, generated_sql, response_text, execution_time_ms)
                VALUES (:uid, :oid, :mode, :qt, :ai, :gsql, :rt, :etms)
                RETURNING id
            """),
            {
                "uid": str(manager_id), "oid": str(org_id), "mode": mode.value,
                "qt": query_text, "ai": "Text-to-SQL via Gemini",
                "gsql": sql, "rt": natural_answer,
                "etms": execution_time_ms,
            },
        )
        await conn.commit()
        inserted_id = ins_r.scalar()

        return {
            "id": str(inserted_id) if inserted_id else "",
            "type": "ai",
            "content": natural_answer,
            "mode": mode.value,
            "sql": sql,
            "executionTime": f"{execution_time_ms}ms",
            "data": rows,
            "rowCount": len(rows),
            "success": True,
        }
