from enum import Enum

class OrgStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    suspended = "suspended"

class UserRole(str, Enum):
    admin = "admin"
    manager = "manager"

class AgentType(str, Enum):
    human = "human"
    bot = "bot"

class ProcessingStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"

class SpeakerRole(str, Enum):
    agent = "agent"
    customer = "customer"

class EventType(str, Enum):
    emotion_shift = "emotion_shift"
    sentiment_drop = "sentiment_drop"
    escalation = "escalation"
    de_escalation = "de_escalation"

class ViolationSeverity(str, Enum):
    minor = "minor"
    critical = "critical"

class FeedbackType(str, Enum):
    emotion_label = "emotion_label"
    score = "score"
    transcription = "transcription"
    compliance = "compliance"
    other = "other"

class QueryMode(str, Enum):
    voice = "voice"
    chat = "chat"
