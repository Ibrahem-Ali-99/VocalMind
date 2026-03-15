from fastapi import APIRouter
from sqlmodel import select, func, case
from sqlalchemy import extract

from app.api.deps import SessionDep
from app.models.interaction import Interaction
from app.models.interaction_score import InteractionScore
from app.models.utterance import Utterance
from app.models.policy import CompanyPolicy, PolicyCompliance
from app.models.user import User as UserModel
from app.models.enums import UserRole, ProcessingStatus
from app.core.cache import dashboard_cache

router = APIRouter()

# Emotion → color mapping (consistent with frontend design)
EMOTION_COLORS = {
    "neutral": "#6B7280",
    "happy": "#10B981",
    "frustrated": "#F59E0B",
    "angry": "#EF4444",
    "sad": "#3B82F6",
    "empathetic": "#8B5CF6",
    "fearful": "#EC4899",
}

# Policy compliance color thresholds
def _compliance_color(rate: float) -> str:
    if rate >= 90:
        return "#10B981"
    if rate >= 80:
        return "#3B82F6"
    if rate >= 70:
        return "#F59E0B"
    return "#EF4444"


@router.get("/stats")
async def get_dashboard_stats(session: SessionDep):
    """Return all data needed by the Manager Dashboard in one call."""
    
    # Check cache first
    cached_data = dashboard_cache.get("manager_stats")
    if cached_data:
        return cached_data

    # 1. KPIs
    kpi_stmt = select(
        func.avg(InteractionScore.overall_score).label("avg_score"),
        func.count(InteractionScore.id).label("total_scored"),
        func.sum(case((InteractionScore.was_resolved == True, 1), else_=0)).label("total_resolved"),  # noqa: E712
    )
    kpi_result = await session.exec(kpi_stmt)
    kpi_row = kpi_result.one()
    avg_score = round(kpi_row.avg_score, 1) if kpi_row.avg_score else 0
    total_scored = kpi_row.total_scored or 0
    total_resolved = kpi_row.total_resolved or 0
    resolution_rate = round((total_resolved / total_scored) * 100, 0) if total_scored else 0

    # 2. Total completed interactions
    total_calls_result = await session.exec(
        select(func.count(Interaction.id)).where(
            Interaction.processing_status == ProcessingStatus.completed
        )
    )
    total_calls = total_calls_result.one_or_none() or 0

    # 3. Policy violations
    violations_result = await session.exec(
        select(func.count(PolicyCompliance.id)).where(
            PolicyCompliance.is_compliant == False  # noqa: E712
        )
    )
    violation_count = violations_result.one_or_none() or 0

    # 4. Weekly Trend
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly_trend_stmt = (
        select(
            extract("dow", Interaction.interaction_date).label("dow"),
            func.avg(InteractionScore.overall_score).label("avg_score"),
        )
        .join(InteractionScore, InteractionScore.interaction_id == Interaction.id)
        .group_by("dow")
        .order_by("dow")
    )
    weekly_trend_result = await session.exec(weekly_trend_stmt)
    
    weekly_trend = []
    for row in weekly_trend_result.all():
        dow = int(row.dow)
        day_index = (dow - 1) % 7 
        day_label = day_names[day_index] if 0 <= day_index < 7 else f"Day{dow}"
        weekly_trend.append({
            "day": day_label,
            "score": round(row.avg_score, 0) if row.avg_score else 0,
        })

    # 5. Emotion Distribution
    emotion_stmt = (
        select(
            Utterance.emotion,
            func.count(Utterance.id).label("count"),
        )
        .where(Utterance.emotion.isnot(None))  # type: ignore[union-attr]
        .group_by(Utterance.emotion)
        .order_by(func.count(Utterance.id).desc())
    )
    emotion_result = await session.exec(emotion_stmt)
    em_rows = emotion_result.all()
    total_emotions = sum(row.count for row in em_rows) if em_rows else 1
    emotion_distribution = [
        {
            "name": (row.emotion or "unknown").capitalize(),
            "value": round((row.count / total_emotions) * 100, 0),
            "color": EMOTION_COLORS.get(row.emotion or "", "#9CA3AF"),
        }
        for row in em_rows
    ]

    # 6. Policy Compliance by Category
    compliance_stmt = (
        select(
            CompanyPolicy.policy_category,
            func.avg(PolicyCompliance.compliance_score).label("avg_rate"),
        )
        .join(CompanyPolicy, CompanyPolicy.id == PolicyCompliance.policy_id)
        .group_by(CompanyPolicy.policy_category)
    )
    compliance_result = await session.exec(compliance_stmt)
    policy_compliance = [
        {
            "category": row.policy_category,
            "rate": round(row.avg_rate * 100, 0) if row.avg_rate else 0,
            "color": _compliance_color(round(row.avg_rate * 100, 0) if row.avg_rate else 0),
        }
        for row in compliance_result.all()
    ]

    # 7. Agent Performance
    agent_perf_stmt = (
        select(
            UserModel.name,
            func.avg(InteractionScore.empathy_score).label("empathy"),
            func.avg(InteractionScore.policy_score).label("policy"),
            func.avg(InteractionScore.resolution_score).label("resolution"),
            func.avg(InteractionScore.overall_score).label("overall"),
        )
        .join(Interaction, Interaction.agent_id == UserModel.id)
        .join(InteractionScore, InteractionScore.interaction_id == Interaction.id)
        .where(UserModel.role == UserRole.agent)
        .group_by(UserModel.id, UserModel.name)
        .order_by(func.avg(InteractionScore.overall_score).desc())
    )
    agent_perf_result = await session.exec(agent_perf_stmt)
    agent_performance = [
        {
            "name": row.name,
            "empathy": round(row.empathy * 10, 0) if row.empathy else 0,
            "policy": round(row.policy * 10, 0) if row.policy else 0,
            "resolution": round(row.resolution * 10, 0) if row.resolution else 0,
            "overallScore": round(row.overall * 10, 0) if row.overall else 0,
            "trend": "up",
        }
        for row in agent_perf_result.all()
    ]

    # 8. Recent Interactions (with violation flag)
    # Using the same logic as baseline but sequential
    violation_subq = (
        select(
            PolicyCompliance.interaction_id,
            func.count(PolicyCompliance.id).label("viol_count"),
        )
        .where(PolicyCompliance.is_compliant == False)  # noqa: E712
        .group_by(PolicyCompliance.interaction_id)
        .subquery()
    )
    recent_stmt = (
        select(
            Interaction.id,
            UserModel.name.label("agent_name"),
            Interaction.interaction_date,
            Interaction.duration_seconds,
            Interaction.language_detected,
            Interaction.has_overlap,
            InteractionScore.overall_score,
            InteractionScore.empathy_score,
            InteractionScore.policy_score,
            InteractionScore.resolution_score,
            InteractionScore.was_resolved,
            func.coalesce(violation_subq.c.viol_count, 0).label("viol_count"),
        )
        .join(UserModel, UserModel.id == Interaction.agent_id)
        .join(InteractionScore, InteractionScore.interaction_id == Interaction.id)
        .outerjoin(violation_subq, violation_subq.c.interaction_id == Interaction.id)
        .order_by(InteractionScore.overall_score.asc())
        .limit(10)
    )
    recent_result = await session.exec(recent_stmt)

    interactions = []
    for row in recent_result.all():
        mins = row.duration_seconds // 60
        secs = row.duration_seconds % 60
        interactions.append({
            "id": str(row.id),
            "agentName": row.agent_name,
            "date": row.interaction_date.strftime("%Y-%m-%d") if row.interaction_date else "",
            "time": row.interaction_date.strftime("%I:%M %p") if row.interaction_date else "",
            "duration": f"{mins}:{secs:02d}",
            "language": row.language_detected or "Unknown",
            "overallScore": round(row.overall_score * 10, 0) if row.overall_score else 0,
            "empathyScore": round(row.empathy_score * 10, 0) if row.empathy_score else 0,
            "policyScore": round(row.policy_score * 10, 0) if row.policy_score else 0,
            "resolutionScore": round(row.resolution_score * 10, 0) if row.resolution_score else 0,
            "resolved": row.was_resolved or False,
            "hasViolation": row.viol_count > 0,
            "hasOverlap": row.has_overlap,
        })

    result = {
        "kpis": {
            "avgScore": avg_score,
            "totalCalls": total_calls,
            "resolutionRate": resolution_rate,
            "violationCount": violation_count,
        },
        "weeklyTrend": weekly_trend,
        "emotionDistribution": emotion_distribution,
        "policyCompliance": policy_compliance,
        "agentPerformance": agent_performance,
        "interactions": interactions,
    }

    # Cache for next time
    dashboard_cache.set("manager_stats", result)

    return result
