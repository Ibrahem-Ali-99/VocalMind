import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta
import random
import re

from sqlmodel import select

from app.core.security import get_password_hash
from app.models.organization import Organization
from app.models.user import User as UserModel
from app.models.enums import UserRole
from app.models.policy import CompanyPolicy, OrganizationPolicy, PolicyCompliance
from app.models.faq import FAQArticle, OrganizationFAQArticle
from app.models.interaction import Interaction
from app.models.interaction_score import InteractionScore
from app.models.enums import ProcessingStatus
from app.models.enums import SpeakerRole
from app.models.utterance import Utterance
from app.models.transcript import Transcript
from app.core.database import engine
from app.core.config import settings
from sqlmodel.ext.asyncio.session import AsyncSession


def parse_nexalink_readme_mapping(readme_path: Path) -> dict[str, str]:
    """Return filename -> agent_name mapping parsed from AudioData README."""
    if not readme_path.exists():
        return {}

    text = readme_path.read_text(encoding="utf-8", errors="ignore")
    sections = re.split(r"\n##\s+", text)
    mapping: dict[str, str] = {}
    for section in sections:
        file_match = re.search(r"\*\*File:\*\*\s*`([^`]+)`", section)
        agent_match = re.search(r"\*\*Agent:\*\*\s*([^|\n]+)", section)
        if not file_match or not agent_match:
            continue

        filename = file_match.group(1).strip()
        agent_name = agent_match.group(1).strip()
        if filename and agent_name:
            mapping[filename] = agent_name
    return mapping


async def create_or_get_organization(session: AsyncSession, name: str, slug: str) -> Organization:
    result = await session.exec(select(Organization).where(Organization.slug == slug))
    org = result.first()
    if not org:
        org = Organization(
            name=name, 
            slug=slug,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            updated_at=datetime.now(timezone.utc).replace(tzinfo=None)
        )
        session.add(org)
        await session.commit()
        await session.refresh(org)
        print(f"Created organization: {name}")
    else:
        print(f"Organization already exists: {name}")
    return org


async def create_or_get_user(session: AsyncSession, org_id, email: str, name: str, role: UserRole) -> UserModel:
    result = await session.exec(select(UserModel).where(UserModel.email == email))
    user = result.first()
    if not user:
        user = UserModel(
            email=email,
            name=name,
            password_hash=get_password_hash("password123"),
            organization_id=org_id,
            role=role
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        print(f"Created user: {email} with role {role}")
    else:
        user.organization_id = org_id
        session.add(user)
        await session.commit()
        print(f"User already exists: {email}, updated org_id")
    return user


async def seed_policies_and_faqs(session: AsyncSession, org_id):
    """Ingest policy and SOP markdown documents into policy/FAQ tables."""
    backend_dir = Path(__file__).resolve().parents[1]
    docs_dir = backend_dir / "sop-standards" / "nexalink" / "parsed-docs"
    
    if not docs_dir.exists():
        print(f"Docs directory not found: {docs_dir}")
        if settings.SEED_MOCK_INTERACTIONS:
            return await seed_dummy_policy(session, org_id)
        return None

    policy_root = docs_dir / "policies"
    sop_root = docs_dir / "sops"

    policy_files = [
        f for f in (policy_root.rglob("*.md") if policy_root.exists() else docs_dir.rglob("*.md"))
        if not any(x in f.name for x in ["_chunks", "_raw"])
        and (not sop_root.exists() or "parsed-docs\\sops" not in str(f))
    ]
    sop_files = [
        f for f in (sop_root.rglob("*.md") if sop_root.exists() else [])
        if not any(x in f.name for x in ["_chunks", "_raw"])
    ]

    if not policy_files and not sop_files:
        print("No markdown files found in parsed-docs directory.")
        if settings.SEED_MOCK_INTERACTIONS:
            return await seed_dummy_policy(session, org_id)
        return None

    print(
        f"Found {len(policy_files)} policy docs and {len(sop_files)} SOP docs in parsed-docs."
    )

    def extract_markdown_title(file_path: Path, text: str) -> str:
        title = file_path.name.replace(".md", "").replace("_", " ").replace("-", " ").title()
        for line in text.splitlines():
            clean_line = line.strip()
            if clean_line.startswith("#"):
                cleaned = (
                    clean_line.lstrip("#")
                    .strip()
                    .replace("POLICY DOCUMENT:", "")
                    .replace("SOP:", "")
                    .strip()
                )
                if cleaned:
                    return cleaned.title()
        return title

    first_policy = None
    policy_count = 0
    faq_count = 0

    for f in policy_files:
        try:
            content = f.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
            continue

        title = extract_markdown_title(f, content)
        category = "Guidelines"

        result = await session.exec(select(CompanyPolicy).where(CompanyPolicy.policy_title == title))
        policy = result.first()

        if not policy:
            policy = CompanyPolicy(
                policy_category=category,
                policy_title=title,
                policy_text=content,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                updated_at=datetime.now(timezone.utc).replace(tzinfo=None)
            )
            session.add(policy)
            await session.commit()
            await session.refresh(policy)
            print(f"Created policy: {title}")
        else:
            if policy.policy_text != content or policy.policy_category != category:
                policy.policy_text = content
                policy.policy_category = category
                policy.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                session.add(policy)
                await session.commit()
                print(f"Updated policy: {title}")

        org_pol_stmt = select(OrganizationPolicy).where(
            OrganizationPolicy.organization_id == org_id,
            OrganizationPolicy.policy_id == policy.id
        )
        org_pol_res = await session.exec(org_pol_stmt)
        if not org_pol_res.first():
            org_policy = OrganizationPolicy(organization_id=org_id, policy_id=policy.id)
            session.add(org_policy)
            await session.commit()

        if not first_policy:
            first_policy = policy
        policy_count += 1

    for f in sop_files:
        try:
            content = f.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading SOP {f.name}: {e}")
            continue

        question = extract_markdown_title(f, content)
        category = "SOP"

        faq_result = await session.exec(select(FAQArticle).where(FAQArticle.question == question))
        faq = faq_result.first()
        if not faq:
            faq = FAQArticle(question=question, answer=content, category=category)
            session.add(faq)
            await session.commit()
            await session.refresh(faq)
            print(f"Created SOP article: {question}")
        else:
            if faq.answer != content or faq.category != category:
                faq.answer = content
                faq.category = category
                faq.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                session.add(faq)
                await session.commit()
                print(f"Updated SOP article: {question}")

        org_faq_stmt = select(OrganizationFAQArticle).where(
            OrganizationFAQArticle.organization_id == org_id,
            OrganizationFAQArticle.article_id == faq.id,
        )
        org_faq_res = await session.exec(org_faq_stmt)
        if not org_faq_res.first():
            session.add(
                OrganizationFAQArticle(
                    organization_id=org_id,
                    article_id=faq.id,
                    is_active=True,
                )
            )
            await session.commit()
        faq_count += 1

    print(f"Organization {org_id} loaded {policy_count} policies and {faq_count} SOP/knowledge docs")

    if not first_policy:
        if settings.SEED_MOCK_INTERACTIONS:
            return await seed_dummy_policy(session, org_id)
        return None

    return first_policy

async def seed_dummy_policy(session: AsyncSession, org_id):
    """Fallback dummy policy if real docs are missing."""
    result = await session.exec(select(CompanyPolicy).where(CompanyPolicy.policy_title == "Nexalink Refund Policy"))
    policy = result.first()
    if not policy:
        policy = CompanyPolicy(
            policy_category="Guidelines",
            policy_title="Nexalink Refund Policy",
            policy_text="Dummy policy text for testing.",
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            updated_at=datetime.now(timezone.utc).replace(tzinfo=None)
        )
        session.add(policy)
        await session.commit()
        await session.refresh(policy)
    
    org_pol_stmt = select(OrganizationPolicy).where(
        OrganizationPolicy.organization_id == org_id,
        OrganizationPolicy.policy_id == policy.id
    )
    org_pol_res = await session.exec(org_pol_stmt)
    if not org_pol_res.first():
        org_policy = OrganizationPolicy(organization_id=org_id, policy_id=policy.id)
        session.add(org_policy)
        await session.commit()
    return policy

async def seed_interactions(
    session: AsyncSession,
    org: Organization,
    agents: list[UserModel],
    manager: UserModel,
    policy: CompanyPolicy | None,
):
    # Scan ../AudioData/nexalink for files
    backend_dir = Path(__file__).resolve().parents[1]
    audio_dir = backend_dir.parent / "AudioData" / "nexalink"
    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        return

    mp3_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
    readme_mapping = parse_nexalink_readme_mapping(audio_dir / "README.md")

    unique_readme_agents = sorted({name for name in readme_mapping.values() if name})
    readme_agent_to_user: dict[str, UserModel] = {}
    for idx, agent_name in enumerate(unique_readme_agents):
        readme_agent_to_user[agent_name] = agents[idx % len(agents)]

    assigned_counts = {agent.email: 0 for agent in agents}
    
    if not mp3_files:
        print("No audio files found in ../AudioData/nexalink")
        return
        
    print(f"Found {len(mp3_files)} audio files. Creating interactions...")
    
    for path in mp3_files:
        filename = str(Path("..") / "AudioData" / "nexalink" / path.name)
        # Check if already exists
        result = await session.exec(
            select(Interaction).where(
                Interaction.organization_id == org.id,
                Interaction.audio_file_path == filename,
            )
        )
        existing_interaction = result.first()

        readme_agent_name = readme_mapping.get(path.name)
        assigned_agent = (
            readme_agent_to_user.get(readme_agent_name)
            if readme_agent_name
            else agents[hash(path.name) % len(agents)]
        )

        if existing_interaction:
            existing_interaction.agent_id = assigned_agent.id
            session.add(existing_interaction)
            await session.commit()
            assigned_counts[assigned_agent.email] += 1
            continue

        interaction = Interaction(
            agent_id=assigned_agent.id,
            uploaded_by=manager.id,
            organization_id=org.id,
            interaction_date=(datetime.now(timezone.utc) - timedelta(days=random.randint(0, 10))).replace(tzinfo=None),
            duration_seconds=random.randint(60, 300),
            language_detected="en",
            file_size_bytes=random.randint(10000, 5000000),
            file_format="mp3",
            has_overlap=False,
            processing_status=ProcessingStatus.completed,
            audio_file_path=filename
        )
        session.add(interaction)
        await session.commit()
        await session.refresh(interaction)

        # Create dummy score
        score = InteractionScore(
            interaction_id=interaction.id,
            overall_score=random.uniform(0.6, 0.95),
            empathy_score=random.uniform(0.6, 1.0),
            policy_score=random.uniform(0.7, 1.0),
            resolution_score=random.uniform(0.5, 1.0),
            was_resolved=random.choice([True, False]),
            avg_response_time_seconds=random.uniform(1.0, 5.0),
            scored_at=datetime.now(timezone.utc).replace(tzinfo=None)
        )
        session.add(score)

        transcript = Transcript(
            interaction_id=interaction.id,
            full_text="Customer asked about order support; agent provided a resolution path.",
            overall_confidence=0.9,
        )
        session.add(transcript)
        await session.flush()

        # Dummy Utterance
        utt = Utterance(
            interaction_id=interaction.id,
            transcript_id=transcript.id,
            speaker_role=SpeakerRole.customer,
            start_time_seconds=0.0,
            end_time_seconds=5.0,
            text="Hello, I have a problem with my order.",
            emotion="neutral",
            emotion_confidence=0.9
        )
        session.add(utt)

        # Dummy Policy compliance
        if policy is not None:
            compliance = PolicyCompliance(
                interaction_id=interaction.id,
                policy_id=policy.id,
                is_compliant=random.choice([True, False]),
                compliance_score=random.uniform(0.3, 0.9),
                llm_reasoning="The agent adequately addressed the policy.",
                evidence_text="Agent mentioned 30 days."
            )
            session.add(compliance)
        
        await session.commit()
        assigned_counts[assigned_agent.email] += 1
        print(f"Created dummy interaction for {filename} -> {assigned_agent.name}")

    print("\nNexalink assignment summary:")
    for agent in agents:
        print(f"  {agent.email}: {assigned_counts[agent.email]} interaction(s)")


async def main():
    async with AsyncSession(engine, expire_on_commit=False) as session:
        print("Starting Database Seed for Nexalink...")
        org = await create_or_get_organization(session, "Nexalink", "nexalink")
        
        manager = await create_or_get_user(session, org.id, "manager@nexalink.com", "Nexalink Manager", UserRole.manager)
        agents = [
            await create_or_get_user(session, org.id, "agent.sara@nexalink.com", "Sara", UserRole.agent),
            await create_or_get_user(session, org.id, "agent.mike@nexalink.com", "Mike", UserRole.agent),
            await create_or_get_user(session, org.id, "agent.rania@nexalink.com", "Rania", UserRole.agent),
        ]

        policy = await seed_policies_and_faqs(session, org.id)

        if settings.SEED_MOCK_INTERACTIONS:
            await seed_interactions(session, org, agents, manager, policy)
        else:
            print("Skipping mock interaction seeding (SEED_MOCK_INTERACTIONS=false).")

        print("Seeding complete! You can login with:")
        print("Manager: manager@nexalink.com / password123")
        print("Agent: agent.sara@nexalink.com / password123")
        print("Agent: agent.mike@nexalink.com / password123")
        print("Agent: agent.rania@nexalink.com / password123")

if __name__ == "__main__":
    asyncio.run(main())
