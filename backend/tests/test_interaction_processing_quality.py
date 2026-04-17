from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlmodel import SQLModel, Session, create_engine, select

import app.models  # noqa: F401
from app.core import interaction_processing as ip
from app.models.emotion_event import EmotionEvent
from app.models.enums import ProcessingStatus, SpeakerRole
from app.models.interaction import Interaction
from app.models.organization import Organization
from app.models.transcript import Transcript
from app.models.utterance import Utterance
from app.models.user import User
from app.models.enums import UserRole


class _SyncResult:
    def __init__(self, result):
        self._result = result

    def first(self):
        return self._result.first()

    def all(self):
        return self._result.all()


class _AsyncSessionAdapter:
    def __init__(self, wrapped: Session):
        self._wrapped = wrapped

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def exec(self, statement):
        return _SyncResult(self._wrapped.exec(statement))

    async def get(self, *args, **kwargs):
        return self._wrapped.get(*args, **kwargs)

    def add(self, instance):
        self._wrapped.add(instance)

    async def flush(self):
        self._wrapped.flush()

    async def commit(self):
        self._wrapped.commit()


@pytest.mark.asyncio
async def test_process_interaction_uses_stable_role_mapping(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'interaction-quality.db'}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    session = Session(engine)

    org_id = uuid4()
    manager_id = uuid4()
    agent_id = uuid4()
    interaction_id = uuid4()
    session.add(
        Organization(
            id=org_id,
            name="Nexa",
            slug="nexa",
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
    )
    session.add(
        User(
            id=manager_id,
            organization_id=org_id,
            email="manager@example.com",
            password_hash="hash",
            name="Manager",
            role=UserRole.manager,
            is_active=True,
        )
    )
    session.add(
        User(
            id=agent_id,
            organization_id=org_id,
            email="agent@example.com",
            password_hash="hash",
            name="Agent",
            role=UserRole.agent,
            is_active=True,
        )
    )
    session.add(
        Interaction(
            id=interaction_id,
            organization_id=org_id,
            agent_id=agent_id,
            uploaded_by=manager_id,
            audio_file_path=str(Path(tmp_path / "call.wav")),
            processing_status=ProcessingStatus.pending,
            duration_seconds=0,
            file_size_bytes=10,
            file_format="wav",
            interaction_date=datetime.now(timezone.utc).replace(tzinfo=None),
        )
    )
    session.add(Transcript(interaction_id=interaction_id, full_text=""))
    session.commit()

    monkeypatch.setattr(ip, "AsyncSession", lambda *args, **kwargs: _AsyncSessionAdapter(session))
    monkeypatch.setattr(ip, "engine", SimpleNamespace())
    monkeypatch.setattr(ip, "evaluate_interaction_triggers", AsyncMock(return_value=None))
    monkeypatch.setattr(ip, "relabel_segments_with_speaker_model", lambda segments: segments)

    async def _fake_fetch(_path):
        return b"RIFF0000WAVEfmt ", "call.wav"

    async def _fake_analyze(_audio, _filename, _content_type):
        return {
            "language": "en",
            "text": "",
            "top_score": 0.88,
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "How can I help you today?",
                    "speaker": "SPEAKER_01",
                    "speaker_meta": {"source": "diarization"},
                    "emotion": "neutral",
                    "emotion_scores": [{"label": "neutral", "score": 0.8}],
                },
                {
                    "start": 1.1,
                    "end": 2.0,
                    "text": "I need help with my account.",
                    "speaker": "SPEAKER_00",
                    "speaker_meta": {"source": "diarization"},
                    "emotion": "frustrated",
                    "emotion_scores": [{"label": "frustrated", "score": 0.9}],
                },
                {
                    "start": 2.1,
                    "end": 3.0,
                    "text": "I still cannot access it.",
                    "speaker": "SPEAKER_00",
                    "speaker_meta": {"source": "diarization"},
                    "emotion": "frustrated",
                    "emotion_scores": [{"label": "frustrated", "score": 0.9}],
                },
            ],
        }

    monkeypatch.setattr(ip, "fetch_audio_bytes", _fake_fetch)
    monkeypatch.setattr(ip.full_client, "analyze_bytes", _fake_analyze)

    await ip.process_interaction(interaction_id)

    utterances = session.exec(
        select(Utterance).where(Utterance.interaction_id == interaction_id).order_by(Utterance.sequence_index)
    ).all()
    assert [u.speaker_role for u in utterances] == [SpeakerRole.agent, SpeakerRole.customer, SpeakerRole.customer]

    emotion_events = session.exec(select(EmotionEvent).where(EmotionEvent.interaction_id == interaction_id)).all()
    assert len(emotion_events) == 1
    assert emotion_events[0].new_emotion == "frustrated"
