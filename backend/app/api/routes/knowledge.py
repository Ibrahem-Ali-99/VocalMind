# Knowledge Base endpoints — policies and FAQs.

from fastapi import APIRouter
from sqlmodel import select

from app.api.deps import SessionDep
from app.models.policy import CompanyPolicy
from app.models.faq import FAQArticle

router = APIRouter()


@router.get("/policies")
async def list_policies(session: SessionDep):
    """List all company policies."""
    result = await session.exec(
        select(CompanyPolicy).order_by(CompanyPolicy.policy_title)
    )
    policies = result.all()

    return [
        {
            "id": str(p.id),
            "title": p.policy_title,
            "category": p.policy_category,
            "content": p.policy_text,
            "preview": p.policy_text[:60] + "..." if len(p.policy_text) > 60 else p.policy_text,
            "lastUpdated": p.updated_at.strftime("%Y-%m-%d") if p.updated_at else "",
            "isActive": p.is_active,
        }
        for p in policies
    ]


@router.get("/faqs")
async def list_faqs(session: SessionDep):
    """List all FAQ articles."""
    result = await session.exec(
        select(FAQArticle).order_by(FAQArticle.question)
    )
    faqs = result.all()

    return [
        {
            "id": str(f.id),
            "question": f.question,
            "answer": f.answer,
            "preview": f.answer[:60] + "..." if len(f.answer) > 60 else f.answer,
            "category": f.category,
            "isActive": f.is_active,
        }
        for f in faqs
    ]
