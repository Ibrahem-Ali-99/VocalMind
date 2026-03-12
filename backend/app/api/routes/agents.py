# Agent endpoints — agent profile and stats.

from fastapi import APIRouter, HTTPException
from sqlmodel import select, func
from uuid import UUID

from app.api.deps import SessionDep
from app.models.user import User as UserModel
from app.models.interaction import Interaction
from app.models.interaction_score import InteractionScore
from app.models.enums import UserRole

router = APIRouter()


@router.get("")
async def list_agents(session: SessionDep):
    """List all agents — used for agent selection when no auth."""
    result = await session.exec(
        select(UserModel).where(UserModel.role == UserRole.agent, UserModel.is_active == True)  # noqa: E712
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
async def get_agent_profile(agent_id: UUID, session: SessionDep):
    """Get agent profile with stats, weekly trend, and recent calls."""

    # Get the agent user
    result = await session.exec(
        select(UserModel).where(UserModel.id == agent_id, UserModel.role == UserRole.agent)
    )
    agent = result.first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Aggregate scores
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
    avg_overall = round(stats.avg_overall * 10, 0) if stats and stats.avg_overall else 0
    avg_empathy = round(stats.avg_empathy * 10, 0) if stats and stats.avg_empathy else 0
    avg_policy = round(stats.avg_policy * 10, 0) if stats and stats.avg_policy else 0
    avg_resolution = round(stats.avg_resolution * 10, 0) if stats and stats.avg_resolution else 0
    avg_response = f"{stats.avg_response:.1f}s" if stats and stats.avg_response else "N/A"

    # Resolution rate
    res_result = await session.exec(
        select(
            func.count(InteractionScore.id),
        )
        .join(Interaction, Interaction.id == InteractionScore.interaction_id)
        .where(Interaction.agent_id == agent_id, InteractionScore.was_resolved == True)  # noqa: E712
    )
    resolved_count = res_result.one_or_none() or 0
    resolution_rate = round((resolved_count / total_calls) * 100, 0) if total_calls else 0

    # Recent calls
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
    recent_rows = recent_result.all()

    recent_calls = [
        {
            "id": str(r.id),
            "date": r.interaction_date.strftime("%Y-%m-%d") if r.interaction_date else "",
            "time": r.interaction_date.strftime("%I:%M %p") if r.interaction_date else "",
            "score": round(r.overall_score * 10, 0) if r.overall_score else 0,
            "duration": f"{r.duration_seconds // 60}:{r.duration_seconds % 60:02d}",
            "language": r.language_detected or "Unknown",
            "resolved": r.was_resolved or False,
            "hasReview": False,
        }
        for r in recent_rows
    ]

    # Weekly trend (last 5 days with interactions)
    from sqlalchemy import extract
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
    weekly_rows = weekly_result.all()

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly_trend = [
        {
            "day": day_names[(int(r.dow) - 1) % 7],
            "score": round(r.avg_score * 10, 0) if r.avg_score else 0,
        }
        for r in weekly_rows
    ]

    return {
        "id": str(agent.id),
        "name": agent.name,
        "role": agent.role.value if agent.role else "agent",
        "totalCalls": total_calls,
        "callsThisWeek": total_calls,  # TODO: filter by current week
        "teamRank": 1,  # TODO: compute rank among agents
        "avgScore": avg_overall,
        "overallScore": avg_overall,
        "empathyScore": avg_empathy,
        "policyScore": avg_policy,
        "resolutionScore": avg_resolution,
        "resolutionRate": resolution_rate,
        "avgResponseTime": avg_response,
        "trend": "up",  # TODO: compute from snapshots
        "weeklyTrend": weekly_trend,
        "recentCalls": recent_calls,
    }
