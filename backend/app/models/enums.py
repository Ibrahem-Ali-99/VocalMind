from enum import Enum


class OrgStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    suspended = "suspended"


class UserRole(str, Enum):
    admin = "admin"
    manager = "manager"
    agent = "agent"
    ai_agent = "ai_agent"


class AgentType(str, Enum):
    human = "human"
    ai = "ai"


class ProcessingStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class SpeakerRole(str, Enum):
    agent = "agent"
    customer = "customer"


class QueryMode(str, Enum):
    voice = "voice"
    chat = "chat"
