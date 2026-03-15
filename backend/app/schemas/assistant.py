from pydantic import BaseModel
from typing import Optional
from app.models.enums import QueryMode

class AssistantQueryRequest(BaseModel):
    query_text: str
    mode: Optional[QueryMode] = QueryMode.chat
