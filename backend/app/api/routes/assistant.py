import time
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from sqlmodel import select, text
from uuid import UUID

logger = logging.getLogger(__name__)

from app.core.config import settings
from google import genai
from google.genai import types

from app.api.deps import SessionDep
from app.models.query import AssistantQuery
from app.models.enums import QueryMode, UserRole
from app.models.user import User as UserModel
from app.schemas.assistant import AssistantQueryRequest

router = APIRouter()

# ---------------------------------------------------------------------------
# Full schema definition for accurate Text-to-SQL generation
# ---------------------------------------------------------------------------
_SCHEMA_DEFINITION = """
### VocalMind Call Center Database Schema

**Table: organizations**
- id UUID PK
- name VARCHAR

**Table: users**
- id UUID PK
- name VARCHAR
- email VARCHAR
- role VARCHAR  -- values: 'manager', 'agent'
- organization_id UUID FK → organizations.id

**Table: interactions**
- id UUID PK
- organization_id UUID FK → organizations.id
- agent_id UUID FK → users.id
- uploaded_by UUID FK → users.id
- audio_file_path VARCHAR
- duration_seconds INTEGER
- interaction_date TIMESTAMP
- processing_status VARCHAR  -- values: 'completed', 'pending', 'failed'
- language_detected VARCHAR
- has_overlap BOOLEAN
- channel_count SMALLINT

**Table: interaction_scores**
- id UUID PK
- interaction_id UUID FK → interactions.id
- overall_score FLOAT  -- range 0.0 – 1.0
- empathy_score FLOAT  -- range 0.0 – 1.0
- policy_score FLOAT   -- range 0.0 – 1.0
- resolution_score FLOAT  -- range 0.0 – 1.0
- was_resolved BOOLEAN

**Table: company_policies**
- id UUID PK
- policy_category VARCHAR
- policy_title VARCHAR
- policy_text TEXT
- is_active BOOLEAN

**Table: organization_policies**
- id UUID PK
- organization_id UUID FK → organizations.id
- policy_id UUID FK → company_policies.id
- is_active BOOLEAN

**Table: policy_compliance**
- id UUID PK
- interaction_id UUID FK → interactions.id
- policy_id UUID FK → company_policies.id
- is_compliant BOOLEAN
- compliance_score FLOAT  -- range 0.0 – 1.0
- llm_reasoning TEXT

**Table: utterances**
- id UUID PK
- interaction_id UUID FK → interactions.id
- speaker VARCHAR  -- values: 'agent', 'customer'
- emotion VARCHAR  -- values: 'neutral', 'happy', 'frustrated', 'angry', 'sad', 'empathetic', 'fearful'
- start_time FLOAT
- end_time FLOAT
"""

# ---------------------------------------------------------------------------
# SQL generation prompt
# ---------------------------------------------------------------------------
def _build_sql_system_prompt(org_id: UUID) -> str:
    return f"""You are an expert PostgreSQL data analyst for VocalMind, a call-center AI monitoring platform.
Given a natural language question, write a single optimized PostgreSQL SELECT query to answer it.

{_SCHEMA_DEFINITION}

### Critical Rules:
1. ALWAYS filter organization scope: use `organization_id = '{str(org_id)}'` on `users` and `interactions` tables.
2. Output ONLY the raw SQL string — no markdown fences (```sql), no explanations, no quotes.
3. If the question is unrelated to the schema or impossible to answer with the data above, output exactly: UNKNOWN
4. SELECT queries only — no INSERT, UPDATE, DELETE, DROP, TRUNCATE, or DDL.
5. Always add LIMIT 50 unless the user explicitly asks for more or uses an aggregate (COUNT, AVG, SUM).
6. When joining interactions ↔ interaction_scores, join on: interaction_scores.interaction_id = interactions.id
7. Scores are stored as 0.0–1.0 floats. Multiply by 10 to get a 0–10 scale if users ask for "out of 10".
8. For agent name lookups use ILIKE for case-insensitive matching.
9. Cast UUIDs to TEXT (::text) when including them in SELECT output.

### Examples:
Q: Who are the top 5 agents by overall score?
A: SELECT u.name, ROUND(AVG(s.overall_score) * 10, 1) AS avg_score FROM users u JOIN interactions i ON i.agent_id = u.id AND i.organization_id = '{str(org_id)}' JOIN interaction_scores s ON s.interaction_id = i.id WHERE u.role = 'agent' GROUP BY u.id, u.name ORDER BY avg_score DESC LIMIT 5

Q: How many policy violations happened this week?
A: SELECT COUNT(*) AS violation_count FROM policy_compliance pc JOIN interactions i ON pc.interaction_id = i.id WHERE pc.is_compliant = false AND i.organization_id = '{str(org_id)}' AND i.interaction_date >= date_trunc('week', now())

Q: What are the most common emotions among customers?
A: SELECT emotion, COUNT(*) AS count FROM utterances u JOIN interactions i ON u.interaction_id = i.id WHERE u.speaker = 'customer' AND i.organization_id = '{str(org_id)}' GROUP BY emotion ORDER BY count DESC LIMIT 10
"""

# ---------------------------------------------------------------------------
# Answer synthesis prompt  
# ---------------------------------------------------------------------------
def _build_synthesis_prompt(user_query: str, sql: str, rows: list) -> str:
    rows_preview = rows[:20]  # Cap to avoid huge prompts
    return f"""You are a helpful business analyst assistant for a call-center manager.
The manager asked: "{user_query}"

To answer this, the following SQL query was executed:
{sql}

The query returned {len(rows)} row(s). Here are the results (up to 20 shown):
{rows_preview}

Write a concise, natural language answer (2–4 sentences) that directly answers the manager's question using the data above.
- Be specific: mention names, numbers, and percentages from the results.
- If the results are empty, say so clearly and suggest why (e.g., no data yet, filters too strict).
- Do not repeat the SQL query.
- Do not add markdown headers or bullet points — write plain paragraph text.
"""


class IntentResolver:
    """LLM-based Text-to-SQL generator with natural language answer synthesis."""

    def __init__(self):
        self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    async def resolve_sql(self, query: str, org_id: UUID) -> Optional[str]:
        """Convert natural language query into a safe PostgreSQL SELECT statement."""
        system_prompt = _build_sql_system_prompt(org_id)
        prompt = f"{system_prompt}\n\nUser Question: {query}\nSQL:"

        try:
            response = await self._client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            sql = (response.text or "").strip()

            # Strip any accidental markdown fences
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
                logger.error(f"Unsafe/invalid SQL generated: {sql}")
                return None

            return sql

        except Exception as exc:
            logger.error(f"SQL generation via Gemini failed: {exc}")
            return None

    async def synthesize_answer(self, user_query: str, sql: str, rows: list) -> str:
        """Use the LLM to produce a human-readable answer from raw SQL results."""
        prompt = _build_synthesis_prompt(user_query, sql, rows)
        try:
            response = await self._client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3),
            )
            return (response.text or "").strip()
        except Exception as exc:
            logger.warning(f"Answer synthesis failed, using fallback: {exc}")
            if rows:
                return f"I found {len(rows)} result(s) for your query."
            return "The query returned no results."


@router.post("/query")
async def process_assistant_query(
    request: AssistantQueryRequest,
    session: SessionDep,
):
    """Process natural language queries from the Manager Assistant."""
    query_text = request.query_text
    mode = request.mode or QueryMode.chat

    # Identify the manager context
    user_result = await session.exec(select(UserModel).where(UserModel.role == UserRole.manager))
    manager = user_result.first()
    if not manager:
        user_result = await session.exec(select(UserModel))
        manager = user_result.first()

    if not manager:
        raise HTTPException(status_code=404, detail="No user found to attribute query")

    org_id = manager.organization_id
    start_time = time.time()
    resolver = IntentResolver()

    # Inject short-term conversational memory (last 5 min)
    last_query_stmt = (
        select(AssistantQuery)
        .where(
            AssistantQuery.user_id == manager.id,
            AssistantQuery.ai_understanding.isnot(None),
        )
        .order_by(AssistantQuery.created_at.desc())
        .limit(1)
    )
    last_query_result = await session.exec(last_query_stmt)
    last_query = last_query_result.first()

    context_prefix = ""
    if last_query and (time.time() - last_query.created_at.timestamp() < 300):
        context_prefix = (
            f"Conversational context — the user's previous question was: "
            f"'{last_query.query_text}'. Use this only if the current question "
            f"references it with pronouns or relative terms (e.g. 'them', 'that agent', 'those violations'). "
        )

    extended_query = context_prefix + query_text

    # Step 1: Generate SQL
    sql = await resolver.resolve_sql(extended_query, org_id)

    if not sql:
        response_text = (
            "I'm sorry, I can't answer that question with the available data. "
            "Please try asking about agents, interactions, scores, compliance, or emotions."
        )
        failed_query = AssistantQuery(
            user_id=manager.id,
            organization_id=org_id,
            query_mode=mode,
            query_text=query_text,
            response_text=response_text,
            execution_time_ms=int((time.time() - start_time) * 1000),
        )
        session.add(failed_query)
        await session.commit()
        return {
            "id": str(failed_query.id),
            "type": "ai",
            "content": response_text,
            "mode": mode.value,
            "success": False,
        }

    # Step 2: Execute SQL
    try:
        exec_start = time.time()
        result = await session.exec(text(sql))
        rows = [dict(row._mapping) for row in result]
        execution_time_ms = int((time.time() - exec_start) * 1000)
    except Exception as exc:
        await session.rollback()
        logger.error(f"SQL execution error — query: {sql!r} — error: {exc}")
        err_response = (
            "I understood your request but encountered a database error while fetching the data. "
            "Consider rephrasing or simplifying your question."
        )
        fallback_query = AssistantQuery(
            user_id=manager.id,
            organization_id=org_id,
            query_mode=mode,
            query_text=query_text,
            response_text=err_response,
            execution_time_ms=int((time.time() - start_time) * 1000),
        )
        session.add(fallback_query)
        await session.commit()
        return {
            "id": str(fallback_query.id),
            "type": "ai",
            "content": err_response,
            "mode": mode.value,
            "sql": sql,
            "success": False,
        }

    # Step 3: Synthesize a natural language answer from results
    natural_answer = await resolver.synthesize_answer(query_text, sql, rows)

    successful_query = AssistantQuery(
        user_id=manager.id,
        organization_id=org_id,
        query_mode=mode,
        query_text=query_text,
        ai_understanding="Text-to-SQL via Gemini",
        generated_sql=sql,
        response_text=natural_answer,
        execution_time_ms=execution_time_ms,
    )
    session.add(successful_query)
    await session.commit()

    return {
        "id": str(successful_query.id),
        "type": "ai",
        "content": natural_answer,
        "mode": mode.value,
        "sql": sql,
        "executionTime": f"{execution_time_ms}ms",
        "data": rows,
        "rowCount": len(rows),
        "success": True,
    }
