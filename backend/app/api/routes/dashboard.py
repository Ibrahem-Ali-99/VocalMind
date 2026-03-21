import logging
from fastapi import APIRouter
import asyncio
from sqlmodel.ext.asyncio.session import AsyncSession
from app.core.database import engine
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

logger = logging.getLogger(__name__)
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

    async def _fetch_kpis():
        async with AsyncSession(engine) as s:
            kpi_stmt = select(
                func.avg(InteractionScore.overall_score).label("avg_score"),
                func.count(InteractionScore.id).label("total_scored"),
                func.sum(case((InteractionScore.was_resolved.is_(True), 1), else_=0)).label("total_resolved"),
            )
            kpi_result = await s.exec(kpi_stmt)
            kpi_row = kpi_result.one()
            avg_score = round(kpi_row.avg_score, 1) if kpi_row.avg_score else 0
            total_scored = kpi_row.total_scored or 0
            total_resolved = kpi_row.total_resolved or 0
            resolution_rate = round((total_resolved / total_scored) * 100, 0) if total_scored else 0
            return avg_score, total_scored, total_resolved, resolution_rate

    async def _fetch_total_calls():
        async with AsyncSession(engine) as s:
            total_calls_result = await s.exec(
                select(func.count(Interaction.id)).where(
                    Interaction.processing_status == ProcessingStatus.completed
                )
            )
            return total_calls_result.one_or_none() or 0

    async def _fetch_violations():
        async with AsyncSession(engine) as s:
            violations_result = await s.exec(
                select(func.count(PolicyCompliance.id)).where(PolicyCompliance.is_compliant.is_(False))
            )
            return violations_result.one_or_none() or 0

    async def _fetch_weekly_trend():
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        async with AsyncSession(engine) as s:
            weekly_trend_stmt = (
                select(
                    extract("dow", Interaction.interaction_date).label("dow"),
                    func.avg(InteractionScore.overall_score).label("avg_score")
                )
                .join(InteractionScore, InteractionScore.interaction_id == Interaction.id)
                .group_by("dow")
                .order_by("dow")
            )
            weekly_trend_result = await s.exec(weekly_trend_stmt)
            weekly_trend = []
            for row in weekly_trend_result.all():
                dow = int(row.dow)
                day_index = (dow - 1) % 7 
                day_label = day_names[day_index] if 0 <= day_index < 7 else f"Day{dow}"
                weekly_trend.append({
                    "day": day_label,
                    "score": round(row.avg_score, 0) if row.avg_score else 0,
                })
            return weekly_trend

    async def _fetch_emotion_distribution():
        async with AsyncSession(engine) as s:
            emotion_stmt = (
                select(
                    Utterance.emotion,
                    func.count(Utterance.id).label("count"),
                )
                .where(Utterance.emotion.isnot(None))
                .group_by(Utterance.emotion)
                .order_by(func.count(Utterance.id).desc())
            )
            emotion_result = await s.exec(emotion_stmt)
            em_rows = emotion_result.all()
            total_emotions = sum(row.count for row in em_rows) if em_rows else 1
            return [
                {
                    "name": (row.emotion or "unknown").capitalize(),
                    "value": round((row.count / total_emotions) * 100, 0),
                    "color": EMOTION_COLORS.get(row.emotion or "", "#9CA3AF"),
                }
                for row in em_rows
            ]

    async def _fetch_policy_compliance():
        async with AsyncSession(engine) as s:
            compliance_stmt = (
                select(
                    CompanyPolicy.policy_category,
                    func.avg(PolicyCompliance.compliance_score).label("avg_rate"),
                )
                .join(CompanyPolicy, CompanyPolicy.id == PolicyCompliance.policy_id)
                .group_by(CompanyPolicy.policy_category)
            )
            compliance_result = await s.exec(compliance_stmt)
            return [
                {
                    "category": row.policy_category,
                    "rate": round(row.avg_rate * 100, 0) if row.avg_rate else 0,
                    "color": _compliance_color(round(row.avg_rate * 100, 0) if row.avg_rate else 0),
                }
                for row in compliance_result.all()
            ]

    async def _fetch_agent_performance():
        async with AsyncSession(engine) as s:
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
            agent_perf_result = await s.exec(agent_perf_stmt)
            return [
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

    async def _fetch_interactions():
        async with AsyncSession(engine) as s:
            violation_subq = (
                select(
                    PolicyCompliance.interaction_id,
                    func.count(PolicyCompliance.id).label("viol_count"),
                )
                .where(PolicyCompliance.is_compliant.is_(False))
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
            recent_result = await s.exec(recent_stmt)
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
            return interactions

    # Run queries concurrently
    tasks = [
        _fetch_kpis(),
        _fetch_total_calls(),
        _fetch_violations(),
        _fetch_weekly_trend(),
        _fetch_emotion_distribution(),
        _fetch_policy_compliance(),
        _fetch_agent_performance(),
        _fetch_interactions(),
    ]
    
    (
        kpis,
        total_calls,
        violation_count,
        weekly_trend,
        emotion_distribution,
        policy_compliance,
        agent_performance,
        interactions
    ) = await asyncio.gather(*tasks)

    avg_score, total_scored, total_resolved, resolution_rate = kpis

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


async def prewarm_dashboard_cache() -> None:
    """Warm the dashboard cache at startup so the first manager request is instantaneous.
    
    This function opens its own AsyncSession independently of FastAPI's DI system,
    making it safe to call from the app lifespan without a request context.
    """
    try:
        logger.info("Dashboard cache pre-warm started...")
        async with AsyncSession(engine) as s:
            # Reuse the same session for all sub-queries to keep it lightweight
            async def _kpis():
                kpi_stmt = select(
                    func.avg(InteractionScore.overall_score).label("avg_score"),
                    func.count(InteractionScore.id).label("total_scored"),
                    func.sum(case((InteractionScore.was_resolved.is_(True), 1), else_=0)).label("total_resolved"),
                )
                r = await s.exec(kpi_stmt)
                row = r.one()
                avg_score = round(row.avg_score, 1) if row.avg_score else 0
                total_scored = row.total_scored or 0
                total_resolved = row.total_resolved or 0
                resolution_rate = round((total_resolved / total_scored) * 100, 0) if total_scored else 0
                return avg_score, total_scored, total_resolved, resolution_rate

            async def _total_calls():
                r = await s.exec(
                    select(func.count(Interaction.id)).where(
                        Interaction.processing_status == ProcessingStatus.completed
                    )
                )
                return r.one_or_none() or 0

            async def _violations():
                r = await s.exec(
                    select(func.count(PolicyCompliance.id)).where(PolicyCompliance.is_compliant.is_(False))
                )
                return r.one_or_none() or 0

            async def _weekly_trend():
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                stmt = (
                    select(
                        extract("dow", Interaction.interaction_date).label("dow"),
                        func.avg(InteractionScore.overall_score).label("avg_score")
                    )
                    .join(InteractionScore, InteractionScore.interaction_id == Interaction.id)
                    .group_by("dow").order_by("dow")
                )
                r = await s.exec(stmt)
                trend = []
                for row in r.all():
                    dow = int(row.dow)
                    day_index = (dow - 1) % 7
                    day_label = day_names[day_index] if 0 <= day_index < 7 else f"Day{dow}"
                    trend.append({"day": day_label, "score": round(row.avg_score, 0) if row.avg_score else 0})
                return trend

            async def _emotion_distribution():
                stmt = (
                    select(Utterance.emotion, func.count(Utterance.id).label("count"))
                    .where(Utterance.emotion.isnot(None))
                    .group_by(Utterance.emotion)
                    .order_by(func.count(Utterance.id).desc())
                )
                r = await s.exec(stmt)
                rows = r.all()
                total = sum(row.count for row in rows) if rows else 1
                return [
                    {
                        "name": (row.emotion or "unknown").capitalize(),
                        "value": round((row.count / total) * 100, 0),
                        "color": EMOTION_COLORS.get(row.emotion or "", "#9CA3AF"),
                    }
                    for row in rows
                ]

            async def _policy_compliance():
                stmt = (
                    select(
                        CompanyPolicy.policy_category,
                        func.avg(PolicyCompliance.compliance_score).label("avg_rate"),
                    )
                    .join(CompanyPolicy, CompanyPolicy.id == PolicyCompliance.policy_id)
                    .group_by(CompanyPolicy.policy_category)
                )
                r = await s.exec(stmt)
                return [
                    {
                        "category": row.policy_category,
                        "rate": round(row.avg_rate * 100, 0) if row.avg_rate else 0,
                        "color": _compliance_color(round(row.avg_rate * 100, 0) if row.avg_rate else 0),
                    }
                    for row in r.all()
                ]

            async def _agent_performance():
                stmt = (
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
                r = await s.exec(stmt)
                return [
                    {
                        "name": row.name,
                        "empathy": round(row.empathy * 10, 0) if row.empathy else 0,
                        "policy": round(row.policy * 10, 0) if row.policy else 0,
                        "resolution": round(row.resolution * 10, 0) if row.resolution else 0,
                        "overallScore": round(row.overall * 10, 0) if row.overall else 0,
                        "trend": "up",
                    }
                    for row in r.all()
                ]

            async def _interactions():
                violation_subq = (
                    select(
                        PolicyCompliance.interaction_id,
                        func.count(PolicyCompliance.id).label("viol_count"),
                    )
                    .where(PolicyCompliance.is_compliant.is_(False))
                    .group_by(PolicyCompliance.interaction_id)
                    .subquery()
                )
                stmt = (
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
                r = await s.exec(stmt)
                result = []
                for row in r.all():
                    mins = row.duration_seconds // 60
                    secs = row.duration_seconds % 60
                    result.append({
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
                return result

            # Run all sub-queries sequentially within the same session
            kpis = await _kpis()
            total_calls = await _total_calls()
            violation_count = await _violations()
            weekly_trend = await _weekly_trend()
            emotion_distribution = await _emotion_distribution()
            policy_compliance = await _policy_compliance()
            agent_performance = await _agent_performance()
            interactions = await _interactions()

        avg_score, total_scored, total_resolved, resolution_rate = kpis
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
        dashboard_cache.set("manager_stats", result)
        logger.info("Dashboard cache pre-warmed successfully.")
    except Exception as exc:
        logger.warning(f"Dashboard cache pre-warm failed (non-fatal): {exc}")
