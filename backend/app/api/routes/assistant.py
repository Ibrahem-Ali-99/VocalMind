import time
import re
from typing import Optional
from fastapi import APIRouter, HTTPException
from sqlmodel import select, text
from uuid import UUID

from app.api.deps import SessionDep
from app.models.query import AssistantQuery
from app.models.enums import QueryMode, UserRole
from app.models.user import User as UserModel
from app.schemas.assistant import AssistantQueryRequest

router = APIRouter()

class IntentResolver:
    """Simple intent resolver to map natural language to SQL/Logic."""
    
    def resolve(self, query: str) -> Optional[str]:
        query = query.lower().strip()
        
        # 1. Resolution Rate Specific (Priority if specific intent detected)
        if re.search(r"resolution.*rate|resolved|resolution", query):
            return "TOP_RESOLUTION"

        # 2. Top performing agents (General)
        if re.search(r"top.*agent|best.*agent|performing.*agent|performance", query):
            return "TOP_AGENTS"
            
        # 3. Policy violations
        if re.search(r"violation|policy.*not.*compliant|non.*compliant|compliance", query):
            return "VIOLATIONS"
            
        # 4. Emotion trends
        if re.search(r"emotion|feeling|sentiment|angry|happy|mood", query):
            return "EMOTION_DISTRIBUTION"
            
        return None

    def get_sql(self, intent: str, org_id: UUID, original_query: str) -> str:
        original_query = original_query.lower()
        
        if intent == "TOP_AGENTS":
            # For general "top" queries, show both metrics for a complete picture
            return f"""
                SELECT u.name, 
                       ROUND(AVG(s.overall_score)::numeric, 1) as avg_score,
                       ROUND((CAST(SUM(CASE WHEN s.was_resolved THEN 1 ELSE 0 END) AS FLOAT) / COUNT(i.id) * 100)::numeric, 1) as res_rate
                FROM users u
                JOIN interactions i ON i.agent_id = u.id
                JOIN interaction_scores s ON s.interaction_id = i.id
                WHERE u.role = 'agent' AND u.organization_id = '{org_id}'
                GROUP BY u.id, u.name
                ORDER BY avg_score DESC
                LIMIT 5
            """
        elif intent == "TOP_RESOLUTION":
            # Determine ordering: default to DESC (highest first) unless "lowest/worst" is requested
            order = "DESC"
            if "lowest" in original_query or "worst" in original_query or "bad" in original_query or "least" in original_query:
                order = "ASC"
                
            return f"""
                SELECT u.name, 
                       ROUND((CAST(SUM(CASE WHEN s.was_resolved THEN 1 ELSE 0 END) AS FLOAT) / COUNT(i.id) * 100)::numeric, 1) as resolution_rate
                FROM users u
                JOIN interactions i ON i.agent_id = u.id
                JOIN interaction_scores s ON s.interaction_id = i.id
                WHERE u.role = 'agent' AND u.organization_id = '{org_id}'
                GROUP BY u.id, u.name
                ORDER BY resolution_rate {order}
                LIMIT 5
            """
        elif intent == "VIOLATIONS":
            return f"""
                SELECT p.policy_title, COUNT(c.id) as violation_count
                FROM policy_compliance c
                JOIN company_policies p ON p.id = c.policy_id
                JOIN interactions i ON i.id = c.interaction_id
                WHERE c.is_compliant = false AND i.organization_id = '{org_id}'
                GROUP BY p.id, p.policy_title
                ORDER BY violation_count DESC
                LIMIT 10
            """
        elif intent == "EMOTION_DISTRIBUTION":
            return """
                SELECT emotion, COUNT(id) as count
                FROM utterances
                WHERE emotion IS NOT NULL
                GROUP BY emotion
                ORDER BY count DESC
            """
        return ""

@router.post("/query")
async def process_assistant_query(
    request: AssistantQueryRequest,
    session: SessionDep
):
    """Process natural language queries from the Manager Assistant."""
    query_text = request.query_text
    mode = request.mode or QueryMode.chat
    
    # Identify context (Manager)
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
    intent = resolver.resolve(query_text)
    
    if not intent:
        # User feedback: The help message should match the actual supported categories accurately.
        response_text = "I'm sorry, I don't know how to query that yet. I can currently help you with agent performance, resolution rates, policy violations, and emotion trends."
        
        failed_query = AssistantQuery(
            user_id=manager.id,
            organization_id=org_id,
            query_mode=mode,
            query_text=query_text,
            response_text=response_text,
            execution_time_ms=int((time.time() - start_time) * 1000)
        )
        session.add(failed_query)
        await session.commit()
        
        return {
            "id": str(failed_query.id),
            "type": "ai",
            "content": response_text,
            "mode": mode.value,
            "success": False
        }

    # Execute SQL for the resolved intent
    sql = resolver.get_sql(intent, org_id, query_text)
    try:
        execution_start = time.time()
        result = await session.exec(text(sql))
        rows = [dict(row._mapping) for row in result]
        execution_time_ms = int((time.time() - execution_start) * 1000)
        
        # Log success
        intent_pretty = intent.lower().replace('_', ' ')
        successful_query = AssistantQuery(
            user_id=manager.id,
            organization_id=org_id,
            query_mode=mode,
            query_text=query_text,
            ai_understanding=intent,
            generated_sql=sql,
            response_text=f"I've found {len(rows)} results for your research into {intent_pretty}.",
            execution_time_ms=execution_time_ms
        )
        session.add(successful_query)
        await session.commit()
        
        return {
            "id": str(successful_query.id),
            "type": "ai",
            "content": successful_query.response_text,
            "mode": mode.value,
            "sql": sql,
            "executionTime": f"{execution_time_ms}ms",
            "data": rows,
            "success": True
        }
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Database error executing assistant query: {str(e)}")
