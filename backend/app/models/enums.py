from enum import Enum


class OrgStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    suspended = "suspended"


class UserRole(str, Enum):
    """v5.1+: Only manager and agent. admin/ai_agent removed from the DB enum."""
    manager = "manager"
    agent = "agent"


class AgentType(str, Enum):
    human = "human"
    ai = "ai"


class ProcessingStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class JobStage(str, Enum):
    diarization = "diarization"
    stt = "stt"
    emotion = "emotion"
    reasoning = "reasoning"
    scoring = "scoring"
    rag_eval = "rag_eval"


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class SpeakerRole(str, Enum):
    agent = "agent"
    customer = "customer"


class QueryMode(str, Enum):
    voice = "voice"
    chat = "chat"


class FeedbackStatus(str, Enum):
    """Shared by emotion_feedback and compliance_feedback (v5.1+)."""
    pending = "pending"
    reviewed = "reviewed"
    applied = "applied"


class PeriodType(str, Enum):
    """Used by agent_performance_snapshots (v5.1+)."""
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
