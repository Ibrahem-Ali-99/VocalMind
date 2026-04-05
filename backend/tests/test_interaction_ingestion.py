"""Integration checks for real-audio interaction ingestion."""

from pathlib import Path
from uuid import UUID

import pytest
from sqlmodel import SQLModel, Session, create_engine, select

from app.api.deps import get_current_user, get_db, get_session
from app.core.config import settings
from app.models.enums import JobStatus, ProcessingStatus, SpeakerRole, UserRole
from app.models.emotion_event import EmotionEvent
from app.models.interaction import Interaction
from app.models.interaction_score import InteractionScore
from app.models.organization import Organization
from app.models.policy import CompanyPolicy, PolicyCompliance
from app.models.processing import ProcessingJob
from app.models.transcript import Transcript
from app.models.utterance import Utterance
from app.models.user import User

TEST_ORG_ID = UUID("8c662e42-9200-493e-aeb2-17d23ac7222a")
TEST_MANAGER_ID = UUID("11111111-1111-1111-1111-111111111111")
TEST_AGENT_ID = UUID("22222222-2222-2222-2222-222222222222")


@pytest.fixture(autouse=True)
def seed_org_and_auth(client, tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'ingestion.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    test_session = Session(engine)

    class AsyncSessionAdapter:
        def __init__(self, wrapped_session):
            self._session = wrapped_session

        async def exec(self, statement):
            return self._session.exec(statement)

        async def get(self, *args, **kwargs):
            return self._session.get(*args, **kwargs)

        def add(self, instance):
            self._session.add(instance)

        def add_all(self, instances):
            self._session.add_all(instances)

        async def flush(self):
            self._session.flush()

        async def commit(self):
            self._session.commit()

        async def refresh(self, instance):
            self._session.refresh(instance)

        async def close(self):
            self._session.close()

    async_session = AsyncSessionAdapter(test_session)

    organization = Organization(
        id=TEST_ORG_ID,
        name="Nexalink",
        slug="nexalink",
    )
    manager_record = User(
        id=TEST_MANAGER_ID,
        organization_id=TEST_ORG_ID,
        email="manager@nexalink.com",
        password_hash="hash",
        name="Manager",
        role=UserRole.manager,
        is_active=True,
    )
    agent_record = User(
        id=TEST_AGENT_ID,
        organization_id=TEST_ORG_ID,
        email="agent@nexalink.com",
        password_hash="hash",
        name="Agent",
        role=UserRole.agent,
        is_active=True,
    )
    test_session.add_all([organization, manager_record, agent_record])
    test_session.commit()

    async def _override_current_user():
        return User(
            id=TEST_MANAGER_ID,
            organization_id=TEST_ORG_ID,
            email="manager@nexalink.com",
            password_hash="hash",
            name="Manager",
            role=UserRole.manager,
            is_active=True,
        )

    async def _override_get_db():
        yield async_session

    async def _override_get_session():
        yield async_session

    client.app.dependency_overrides[get_current_user] = _override_current_user
    client.app.dependency_overrides[get_db] = _override_get_db
    client.app.dependency_overrides[get_session] = _override_get_session
    monkeypatch.setattr(settings, "LOCAL_AUDIO_STORAGE_DIR", str(tmp_path / "uploads"))

    async def _noop_enqueue(interaction_id):
        return None

    monkeypatch.setattr("app.api.routes.interactions.enqueue_interaction_processing", _noop_enqueue)

    yield test_session

    client.app.dependency_overrides.pop(get_current_user, None)
    client.app.dependency_overrides.pop(get_db, None)
    client.app.dependency_overrides.pop(get_session, None)
    test_session.close()


def test_upload_interaction_creates_pending_job_rows(client, seed_org_and_auth):
    response = client.post(
        "/api/v1/interactions",
        data={"agent_id": str(TEST_AGENT_ID)},
        files={"file": ("call.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()

    interaction_id = UUID(payload["interactionId"])
    session = seed_org_and_auth
    interaction = session.get(Interaction, interaction_id)
    transcript = session.exec(
        select(Transcript).where(Transcript.interaction_id == interaction_id)
    ).first()
    jobs = session.exec(
        select(ProcessingJob).where(ProcessingJob.interaction_id == interaction_id)
    ).all()

    assert interaction is not None
    assert interaction.organization_id == TEST_ORG_ID
    assert interaction.agent_id == TEST_AGENT_ID
    assert interaction.uploaded_by == TEST_MANAGER_ID
    assert interaction.processing_status == ProcessingStatus.pending
    assert interaction.audio_file_path
    assert Path(interaction.audio_file_path).exists()

    assert transcript is not None
    assert transcript.full_text == ""

    assert len(jobs) == 6
    assert {job.stage.value for job in jobs} == {"diarization", "stt", "emotion", "reasoning", "scoring", "rag_eval"}
    assert all(job.status.value == "pending" for job in jobs)


def test_reprocess_resets_artifacts_and_jobs(client, seed_org_and_auth):
    create_response = client.post(
        "/api/v1/interactions",
        data={"agent_id": str(TEST_AGENT_ID)},
        files={"file": ("call.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
    )
    assert create_response.status_code == 200, create_response.text

    interaction_id = UUID(create_response.json()["interactionId"])
    session = seed_org_and_auth

    interaction = session.get(Interaction, interaction_id)
    interaction.processing_status = ProcessingStatus.completed
    session.add(interaction)

    transcript = session.exec(
        select(Transcript).where(Transcript.interaction_id == interaction_id)
    ).first()
    transcript.full_text = "processed transcript"
    transcript.overall_confidence = 0.99
    session.add(transcript)

    utterance = Utterance(
        interaction_id=interaction_id,
        transcript_id=transcript.id,
        sequence_index=0,
        start_time_seconds=0.0,
        end_time_seconds=1.0,
        text="hello",
        emotion="neutral",
        emotion_confidence=0.8,
    )
    session.add(utterance)
    session.flush()

    policy = CompanyPolicy(
        policy_category="Guidelines",
        policy_title="Test Policy",
        policy_text="Test policy text",
    )
    session.add(policy)
    session.flush()

    session.add(
        EmotionEvent(
            interaction_id=interaction_id,
            utterance_id=utterance.id,
            previous_emotion="neutral",
            new_emotion="happy",
            emotion_delta=1.0,
            speaker_role=SpeakerRole.customer,
            jump_to_seconds=1.0,
            confidence_score=0.8,
        )
    )
    session.add(
        InteractionScore(
            interaction_id=interaction_id,
            overall_score=0.8,
            empathy_score=0.8,
            policy_score=0.8,
            resolution_score=0.8,
            total_silence_seconds=0.0,
            avg_response_time_seconds=1.2,
            was_resolved=True,
        )
    )
    session.add(
        PolicyCompliance(
            interaction_id=interaction_id,
            policy_id=policy.id,
            is_compliant=True,
            compliance_score=0.9,
        )
    )

    jobs = session.exec(
        select(ProcessingJob).where(ProcessingJob.interaction_id == interaction_id)
    ).all()
    for job in jobs:
        job.status = JobStatus.completed
        session.add(job)

    session.commit()

    response = client.post(f"/api/v1/interactions/{interaction_id}/reprocess")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["queued"] is True
    assert payload["status"] == "pending"

    updated_interaction = session.get(Interaction, interaction_id)
    assert updated_interaction.processing_status == ProcessingStatus.pending

    updated_transcript = session.exec(
        select(Transcript).where(Transcript.interaction_id == interaction_id)
    ).first()
    assert updated_transcript.full_text == ""
    assert updated_transcript.overall_confidence is None

    assert session.exec(select(Utterance).where(Utterance.interaction_id == interaction_id)).first() is None
    assert session.exec(select(EmotionEvent).where(EmotionEvent.interaction_id == interaction_id)).first() is None
    assert session.exec(select(InteractionScore).where(InteractionScore.interaction_id == interaction_id)).first() is None
    assert session.exec(select(PolicyCompliance).where(PolicyCompliance.interaction_id == interaction_id)).first() is None

    updated_jobs = session.exec(
        select(ProcessingJob).where(ProcessingJob.interaction_id == interaction_id)
    ).all()
    assert len(updated_jobs) == 6
    assert all(job.status.value == "pending" for job in updated_jobs)
