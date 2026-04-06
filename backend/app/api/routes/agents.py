from fastapi import APIRouter, HTTPException
from sqlmodel import select, func
from uuid import UUID
from sqlalchemy import extract
from datetime import datetime, timedelta

from app.api.deps import SessionDep, CurrentUser
from app.models.user import User as UserModel
from app.models.interaction import Interaction
from app.models.interaction_score import InteractionScore
from app.models.enums import UserRole
from app.core.cache import dashboard_cache

router = APIRouter()


@router.get("")
async def list_agents(session: SessionDep, current_user: CurrentUser):
    """List all agents for the current organization."""
    result = await session.exec(
        select(UserModel).where(
            UserModel.role == UserRole.agent, 
            UserModel.is_active == True,  # noqa: E712
            UserModel.organization_id == current_user.organization_id
        )
    )
    agents = result.all()
    return [
        {
            "id": str(a.id),
            "name": a.name,
            "role": a.role.value if a.role else "agent",
        }
        for a in agents
    ]


@router.get("/{agent_id}")
async def get_agent_profile(agent_id: UUID, session: SessionDep, current_user: CurrentUser):
    """Get agent profile with stats, weekly trend, and recent calls."""

    # Check cache first
    cache_key = f"agent_profile_{current_user.organization_id}_{agent_id}"
    cached_data = dashboard_cache.get(cache_key)
    if cached_data:
        return cached_data

    # 1. Get the agent user
    result = await session.exec(
        select(UserModel).where(
            UserModel.id == agent_id, 
            UserModel.role == UserRole.agent,
            UserModel.organization_id == current_user.organization_id
        )
    )
    agent = result.first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # 2. Aggregate scores
    scores_stmt = (
        select(
            func.count(Interaction.id).label("total_calls"),
            func.avg(InteractionScore.overall_score).label("avg_overall"),
            func.avg(InteractionScore.empathy_score).label("avg_empathy"),
            func.avg(InteractionScore.policy_score).label("avg_policy"),
            func.avg(InteractionScore.resolution_score).label("avg_resolution"),
            func.avg(InteractionScore.avg_response_time_seconds).label("avg_response"),
        )
        .join(InteractionScore, InteractionScore.interaction_id == Interaction.id)
        .where(Interaction.agent_id == agent_id)
    )
    scores_result = await session.exec(scores_stmt)
    stats = scores_result.first()

    total_calls = stats.total_calls if stats and stats.total_calls else 0
    avg_overall = round(stats.avg_overall * 100, 0) if stats and stats.avg_overall else 0
    avg_empathy = round(stats.avg_empathy * 100, 0) if stats and stats.avg_empathy else 0
    avg_policy = round(stats.avg_policy * 100, 0) if stats and stats.avg_policy else 0
    avg_resolution = round(stats.avg_resolution * 100, 0) if stats and stats.avg_resolution else 0
    avg_response = f"{stats.avg_response:.1f}s" if stats and stats.avg_response else "N/A"

    # 3. Resolution rate
    res_result = await session.exec(
        select(func.count(InteractionScore.id))
        .join(Interaction, Interaction.id == InteractionScore.interaction_id)
        .where(Interaction.agent_id == agent_id, InteractionScore.was_resolved == True)  # noqa: E712
    )
    resolved_count = res_result.one_or_none() or 0
    resolution_rate = round((resolved_count / total_calls) * 100, 0) if total_calls else 0

    # 4. Recent calls
    recent_stmt = (
        select(
            Interaction.id,
            Interaction.interaction_date,
            Interaction.duration_seconds,
            Interaction.language_detected,
            InteractionScore.overall_score,
            InteractionScore.was_resolved,
        )
        .outerjoin(InteractionScore, InteractionScore.interaction_id == Interaction.id)
        .where(Interaction.agent_id == agent_id)
        .order_by(Interaction.interaction_date.desc())
        .limit(10)
    )
    recent_result = await session.exec(recent_stmt)
    recent_calls = [
        {
            "id": str(r.id),
            "date": r.interaction_date.strftime("%Y-%m-%d") if r.interaction_date else "",
            "time": r.interaction_date.strftime("%I:%M %p") if r.interaction_date else "",
            "score": round(r.overall_score * 100, 0) if r.overall_score else 0,
            "duration": f"{r.duration_seconds // 60}:{r.duration_seconds % 60:02d}",
            "language": r.language_detected or "Unknown",
            "resolved": r.was_resolved or False,
            "hasReview": False,
        }
        for r in recent_result.all()
    ]

    # 5. Weekly trend
    weekly_stmt = (
        select(
            extract("dow", Interaction.interaction_date).label("dow"),
            func.avg(InteractionScore.overall_score).label("avg_score"),
        )
        .join(InteractionScore, InteractionScore.interaction_id == Interaction.id)
        .where(Interaction.agent_id == agent_id)
        .group_by("dow")
        .order_by("dow")
        .limit(7)
    )
    weekly_result = await session.exec(weekly_stmt)
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly_trend = [
        {
            "day": day_names[(int(r.dow) - 1) % 7],
            "score": round(r.avg_score * 100, 0) if r.avg_score else 0,
        }
        for r in weekly_result.all()
    ]

    # 6. Calls this week
    start_of_week = datetime.now() - timedelta(days=datetime.now().weekday())
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    calls_this_week_res = await session.exec(
        select(func.count(Interaction.id))
        .where(Interaction.agent_id == agent_id, Interaction.interaction_date >= start_of_week)
    )
    calls_this_week = calls_this_week_res.one_or_none() or 0

    result = {
        "id": str(agent.id),
        "name": agent.name,
        "role": agent.role.value if agent.role else "agent",
        "totalCalls": total_calls,
        "callsThisWeek": calls_this_week,
        "teamRank": 1, 
        "avgScore": avg_overall,
        "overallScore": avg_overall,
        "empathyScore": avg_empathy,
        "policyScore": avg_policy,
        "resolutionScore": avg_resolution,
        "resolutionRate": resolution_rate,
        "avgResponseTime": avg_response,
        "trend": "up",
        "weeklyTrend": weekly_trend,
        "recentCalls": recent_calls,
    }

    # Cache result
    dashboard_cache.set(cache_key, result)

    return result
