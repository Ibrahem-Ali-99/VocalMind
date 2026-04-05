from __future__ import annotations

import asyncio
from uuid import UUID

from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.database import engine
from app.models.emotion_event import EmotionEvent
from app.models.interaction import Interaction
from app.models.interaction_score import InteractionScore
from app.models.organization import Organization
from app.models.policy import PolicyCompliance
from app.models.processing import ProcessingJob
from app.models.transcript import Transcript
from app.models.utterance import Utterance


async def purge_seeded_interactions(org_slug: str = "nexalink") -> None:
    async with AsyncSession(engine, expire_on_commit=False) as session:
        org = (await session.exec(select(Organization).where(Organization.slug == org_slug))).first()
        if not org:
            print(f"Organization not found: {org_slug}")
            return

        interactions = (
            await session.exec(
                select(Interaction.id).where(
                    Interaction.organization_id == org.id,
                    (
                        Interaction.audio_file_path.like("%%AudioData%%nexalink%%")
                    ),
                )
            )
        ).all()

        interaction_ids = [UUID(str(i)) for i in interactions]
        if not interaction_ids:
            print("No seeded/mock interactions found to purge.")
            return

        await session.exec(delete(EmotionEvent).where(EmotionEvent.interaction_id.in_(interaction_ids)))
        await session.exec(delete(Utterance).where(Utterance.interaction_id.in_(interaction_ids)))
        await session.exec(delete(PolicyCompliance).where(PolicyCompliance.interaction_id.in_(interaction_ids)))
        await session.exec(delete(InteractionScore).where(InteractionScore.interaction_id.in_(interaction_ids)))
        await session.exec(delete(ProcessingJob).where(ProcessingJob.interaction_id.in_(interaction_ids)))
        await session.exec(delete(Transcript).where(Transcript.interaction_id.in_(interaction_ids)))
        await session.exec(delete(Interaction).where(Interaction.id.in_(interaction_ids)))

        # Keep org-level knowledge links intact. This script only purges seeded calls.
        await session.commit()
        print(f"Purged {len(interaction_ids)} seeded/mock interactions for org '{org_slug}'.")


if __name__ == "__main__":
    asyncio.run(purge_seeded_interactions())
