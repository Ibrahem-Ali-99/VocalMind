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
# Help response
# ---------------------------------------------------------------------------
_HELP_RESPONSE = """Here is what you can ask me about your call center:\n\n**Agents & Performance**\n- "Who are the top 5 agents by overall score?"\n- "Which agent has the lowest resolution rate?"\n- "Show agents ranked by empathy score"\n\n**Interactions & Calls**\n- "How many calls were not resolved?"\n- "Show interactions in the last 30 days"\n- "Which calls had the highest empathy score?"\n\n**Policy Violations**\n- "List all policy violations"\n- "Which agents violated the Escalation Protocol?"\n\n**Customer Emotions**\n- "What are the most common customer emotions?"\n- "Show calls where the customer was frustrated"\n\n**Available score columns:**\noverall_score, empathy_score, policy_score, resolution_score (all 0-10 scale)\n\nTip: you can add time filters like "last 30 days", "last 3 months", or "all time"."""

_HELP_TRIGGERS = {"help", "?", "what can i ask", "what can you do", "commands", "guide"}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_SCHEMA = """
Tables and columns:
- users: id(UUID), name, email, role('manager'/'agent'), organization_id(UUID)
- organizations: id(UUID), name
- interactions: id(UUID), organization_id(UUID), agent_id(UUID->users.id), duration_seconds, interaction_date(TIMESTAMP), processing_status('completed'/'pending'/'failed'), language_detected, has_overlap(BOOL)
- interaction_scores: id(UUID), interaction_id(UUID->interactions.id), overall_score(FLOAT 0-1), empathy_score(FLOAT 0-1), policy_score(FLOAT 0-1), resolution_score(FLOAT 0-1), was_resolved(BOOL)
- policy_compliance: id(UUID), interaction_id(UUID), policy_id(UUID->company_policies.id), is_compliant(BOOL), compliance_score(FLOAT 0-1), llm_reasoning(TEXT)
- company_policies: id(UUID), policy_category, policy_title, policy_text, is_active(BOOL)
- utterances: id(UUID), interaction_id(UUID), speaker('agent'/'customer'), emotion('neutral'/'happy'/'frustrated'/'angry'/'sad'/'empathetic'/'fearful'), start_time(FLOAT), end_time(FLOAT)
"""


def _build_sql_prompt(org_id: UUID, question: str) -> str:
    org = str(org_id)
    return f"""You are a PostgreSQL expert for a call-center analytics platform.
Convert the manager's question into a single valid PostgreSQL SELECT query.

Schema:
{_SCHEMA}

Rules:
1. Always restrict to the organization: use organization_id = '{org}' on `users` OR `interactions` tables.
2. Return ONLY raw SQL - no markdown, no explanation.
3. SELECT only - never DELETE/UPDATE/DROP/INSERT/ALTER/CREATE.
4. Add LIMIT 50 unless the user asks for more or uses aggregates (COUNT/SUM/AVG).
5. Float casting: PostgreSQL ROUND() requires NUMERIC type. Always write ROUND(expr::NUMERIC, 1).
6. Scores in db are 0.0-1.0. To show as 0-10: multiply by 10. To show as percent: multiply by 100.
7. For ambiguous "top/best" keywords, use `overall_score` as the primary metric.
8. Time shortcuts: "this week" -> interaction_date >= date_trunc('week', now()), "last 30 days" -> interaction_date >= NOW() - INTERVAL '30 days'. If no time is specified, query all data.
9. Match names using ILIKE. Cast UUID output with ::text.
10. Join interaction_scores on: interaction_scores.interaction_id = interactions.id

Examples:
Q: Who are the top 5 agents by overall score?
A: SELECT u.name, ROUND(AVG(s.overall_score)::NUMERIC * 10, 1) AS avg_score FROM users u JOIN interactions i ON i.agent_id = u.id AND i.organization_id = '{org}' JOIN interaction_scores s ON s.interaction_id = i.id WHERE u.role = 'agent' GROUP BY u.id, u.name ORDER BY avg_score DESC LIMIT 5

Q: Show top performing agents
A: SELECT u.name, ROUND(AVG(s.overall_score)::NUMERIC * 10, 1) AS avg_score, ROUND(AVG(s.empathy_score)::NUMERIC * 10, 1) AS empathy FROM users u JOIN interactions i ON i.agent_id = u.id AND i.organization_id = '{org}' JOIN interaction_scores s ON s.interaction_id = i.id WHERE u.role = 'agent' GROUP BY u.id, u.name ORDER BY avg_score DESC LIMIT 10

Q: Which agent has the highest empathy score?
A: SELECT u.name, ROUND(AVG(s.empathy_score)::NUMERIC * 10, 1) AS avg_empathy FROM users u JOIN interactions i ON i.agent_id = u.id AND i.organization_id = '{org}' JOIN interaction_scores s ON s.interaction_id = i.id WHERE u.role = 'agent' GROUP BY u.id, u.name ORDER BY avg_empathy DESC LIMIT 1

Q: Show top performing agents this week
A: SELECT u.name, ROUND(AVG(s.overall_score)::NUMERIC * 10, 1) AS avg_score FROM users u JOIN interactions i ON i.agent_id = u.id AND i.organization_id = '{org}' JOIN interaction_scores s ON s.interaction_id = i.id WHERE u.role = 'agent' AND i.interaction_date >= date_trunc('week', now()) GROUP BY u.id, u.name ORDER BY avg_score DESC LIMIT 10

Q: agent with lowest resolution rate
A: SELECT u.name, ROUND(AVG(s.resolution_score)::NUMERIC * 100, 1) AS resolution_pct FROM users u JOIN interactions i ON i.agent_id = u.id AND i.organization_id = '{org}' JOIN interaction_scores s ON s.interaction_id = i.id WHERE u.role = 'agent' GROUP BY u.id, u.name ORDER BY resolution_pct ASC LIMIT 1

Q: list policy violations
A: SELECT i.id::text AS interaction_id, cp.policy_title, ROUND(pc.compliance_score::NUMERIC * 10, 1) AS compliance_score, pc.llm_reasoning FROM policy_compliance pc JOIN company_policies cp ON pc.policy_id = cp.id JOIN interactions i ON pc.interaction_id = i.id WHERE pc.is_compliant = false AND i.organization_id = '{org}' LIMIT 50

Q: most common customer emotions
A: SELECT emotion, COUNT(*) AS count FROM utterances u JOIN interactions i ON u.interaction_id = i.id WHERE u.speaker = 'customer' AND i.organization_id = '{org}' GROUP BY emotion ORDER BY count DESC LIMIT 10

Question: {question}
SQL:"""


def _build_synthesis_prompt(question: str, sql: str, rows: list) -> str:
    return f"""You are a concise data analyst for a call-center manager.
The manager asked: "{question}"

The following SQL was executed and returned {len(rows)} rows (showing up to 20):
{rows[:20]}

Write 2-4 sentences of plain text answering the question using the actual data.
- Mention specific names and numbers from the results.
- Scores in the data are already scaled: if a value looks like 8.5 it is out of 10; if it looks like 85 it is a percentage.
- If a raw 0-1 float appears (like 0.8), present it as 80%.
- If results are empty, say no data was found for that time period and suggest trying "last 30 days" or "all time".
- No markdown, no bullet points, no SQL repetition."""


import random

class IntentResolver:
    def __init__(self):
        # Pool of Gemini API keys to avoid 15 RPM free tier limits
        api_keys = [
            settings.GOOGLE_API_KEY,  # Original key
        ]
        self._keys = [k for k in api_keys if k]
        # We don't initialize a single client here anymore

    async def _generate_content_with_fallback(self, prompt: str, temperature: float) -> Optional[str]:
        """Tries to generate content using available keys, falling back on 429 errors."""
        if not self._keys:
            logger.error("No valid Gemini API keys configured.")
            raise Exception("No API keys")
            
        keys_to_try = list(self._keys)
        random.shuffle(keys_to_try)
        
        last_error = None
        for key in keys_to_try:
            try:
                client = genai.Client(api_key=key)
                response = await client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature),
                )
                return response.text
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    # Rate limit hit with this key, try the next one
                    last_error = exc
                    continue
                else:
                    # Other errors (auth, bad request) should break immediately
                    raise exc
                    
        # If we exhausted all keys and all gave 429s, raise the last rate limit error
        if last_error:
            raise last_error
        return None

    async def resolve_sql(self, question: str, org_id: UUID) -> Optional[str]:
        prompt = _build_sql_prompt(org_id, question)
        try:
            response_text = await self._generate_content_with_fallback(prompt, 0.0)
            sql = (response_text or "").strip()

            # Strip markdown fences if Gemini adds them
            for fence in ("```sql", "```"):
                if sql.startswith(fence):
                    sql = sql[len(fence):]
            if sql.endswith("```"):
                sql = sql[:-3]
            sql = sql.strip()

            if not sql or "UNKNOWN" in sql.upper() or not sql.lower().startswith("select"):
                return None

            safe = sql.lower()
            bad = ("drop ", "delete ", "update ", "insert ", "truncate ", "alter ", "create ")
            if any(k in safe for k in bad):
                logger.error(f"Rejected unsafe SQL: {sql[:200]}")
                return None

            return sql
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                logger.error(f"Gemini API Rate Limit hit after trying all keys: {exc}")
                return "RATE_LIMIT_ERROR"
            logger.error(f"Gemini SQL gen failed: {exc}")
            return None

    async def synthesize_answer(self, question: str, sql: str, rows: list) -> str:
        prompt = _build_synthesis_prompt(question, sql, rows)
        try:
            response_text = await self._generate_content_with_fallback(prompt, 0.3)
            return (response_text or "").strip()
        except Exception as exc:
            logger.warning(f"Synthesis failed: {exc}")
            return f"I found {len(rows)} result(s)." if rows else "No results found for that query."

@router.get("/history")
async def get_assistant_history(session: SessionDep):
    """Retrieve the recent chat history for the manager."""
    async with engine.connect() as conn:
        r = await conn.execute(
            text("SELECT id, organization_id FROM users WHERE role = 'manager' LIMIT 1")
        )
        row = r.first()
        if not row:
            r = await conn.execute(text("SELECT id, organization_id FROM users LIMIT 1"))
            row = r.first()
        if not row:
            return []
            
        manager_id, _ = row
        
        hist_r = await conn.execute(
            text("SELECT id, query_text, response_text, created_at FROM assistant_queries WHERE user_id = :uid ORDER BY created_at ASC LIMIT 50"),
            {"uid": str(manager_id)}
        )
        
        history = []
        for h in hist_r.all():
            idx, q, r_text, _ = h
            if not r_text:
                continue
            history.append({"id": f"q_{idx}", "type": "user", "content": q, "mode": "chat"})
            history.append({"id": f"a_{idx}", "type": "ai", "content": r_text, "mode": "chat", "success": True})
            
        return history


@router.post("/query")
async def process_assistant_query(
    request: AssistantQueryRequest,
    session: SessionDep,
):
    query_text = request.query_text.strip()
    mode = request.mode or QueryMode.chat

    # Help / schema discovery
    if query_text.lower() in _HELP_TRIGGERS or query_text in ("?", ""):
        return {"id": "", "type": "ai", "content": _HELP_RESPONSE, "mode": mode.value, "success": True}

    # Use raw engine connection -- bypasses SQLModel ORM lazy-loading (MissingGreenlet) bug
    async with engine.connect() as conn:

        # Find manager
        r = await conn.execute(
            text("SELECT id, organization_id FROM users WHERE role = 'manager' LIMIT 1")
        )
        row = r.first()
        if not row:
            r = await conn.execute(text("SELECT id, organization_id FROM users LIMIT 1"))
            row = r.first()
        if not row:
            raise HTTPException(status_code=404, detail="No user found")

        manager_id, org_id = row
        start_time = time.time()
        resolver = IntentResolver()

        # Conversational context (last 5 min)
        ctx_r = await conn.execute(
            text("SELECT query_text, created_at FROM assistant_queries WHERE user_id = :uid AND ai_understanding IS NOT NULL ORDER BY created_at DESC LIMIT 1"),
            {"uid": str(manager_id)},
        )
        ctx_row = ctx_r.first()
        context_note = ""
        if ctx_row and (time.time() - ctx_row[1].timestamp() < 300):
            context_note = f"[Previous question for context: \"{ctx_row[0]}\". Use it only if current question references it with words like 'them', 'that', 'same', 'highest one', etc.] "

        question = context_note + query_text

        # Generate SQL
        sql = await resolver.resolve_sql(question, org_id)

        if sql == "RATE_LIMIT_ERROR":
            msg = "I'm currently receiving too many requests. Google Gemini rate limits have been temporarily exceeded. Please try again in a few minutes."
            await conn.execute(
                text("INSERT INTO assistant_queries (user_id, organization_id, query_mode, query_text, response_text, execution_time_ms) VALUES (:u, :o, :m, :q, :r, :e)"),
                {"u": str(manager_id), "o": str(org_id), "m": mode.value, "q": query_text, "r": msg, "e": int((time.time()-start_time)*1000)},
            )
            await conn.commit()
            return {"type": "ai", "content": msg, "mode": mode.value, "success": False}

        if not sql:
            msg = ("I'm not sure how to answer that from the available data. "
                   "Try asking about agents, scores, violations, or emotions -- or type 'help'.")
            await conn.execute(
                text("INSERT INTO assistant_queries (user_id, organization_id, query_mode, query_text, response_text, execution_time_ms) VALUES (:u, :o, :m, :q, :r, :e)"),
                {"u": str(manager_id), "o": str(org_id), "m": mode.value, "q": query_text, "r": msg, "e": int((time.time()-start_time)*1000)},
            )
            await conn.commit()
            return {"type": "ai", "content": msg, "mode": mode.value, "success": False}

        # Execute SQL
        try:
            t0 = time.time()
            res = await conn.execute(text(sql))
            rows = [dict(r._mapping) for r in res]
            exec_ms = int((time.time() - t0) * 1000)
        except Exception as exc:
            await conn.rollback()
            logger.error(f"SQL exec error: {exc} | SQL: {sql[:300]}")
            err_msg = "I understood your request but hit a database error. Try rephrasing or type 'help' for example queries."
            await conn.execute(
                text("INSERT INTO assistant_queries (user_id, organization_id, query_mode, query_text, response_text, execution_time_ms) VALUES (:u, :o, :m, :q, :r, :e)"),
                {"u": str(manager_id), "o": str(org_id), "m": mode.value, "q": query_text, "r": err_msg, "e": int((time.time()-start_time)*1000)},
            )
            await conn.commit()
            return {"type": "ai", "content": err_msg, "mode": mode.value, "sql": sql, "success": False}

        # Synthesize answer
        answer = await resolver.synthesize_answer(query_text, sql, rows)

        ins = await conn.execute(
            text("INSERT INTO assistant_queries (user_id, organization_id, query_mode, query_text, ai_understanding, generated_sql, response_text, execution_time_ms) VALUES (:u, :o, :m, :q, :ai, :s, :r, :e) RETURNING id"),
            {"u": str(manager_id), "o": str(org_id), "m": mode.value, "q": query_text, "ai": "Gemini Text-to-SQL", "s": sql, "r": answer, "e": exec_ms},
        )
        await conn.commit()
        qid = ins.scalar()

        return {
            "id": str(qid) if qid else "",
            "type": "ai",
            "content": answer,
            "mode": mode.value,
            "sql": sql,
            "executionTime": f"{exec_ms}ms",
            "data": rows,
            "rowCount": len(rows),
            "success": True,
        }
