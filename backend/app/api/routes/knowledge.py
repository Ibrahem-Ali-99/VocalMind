from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select
from pypdf import PdfReader

from app.api.deps import SessionDep, CurrentUser
from app.core.config import settings
from app.models.organization import Organization
from app.models.policy import CompanyPolicy, OrganizationPolicy, PolicyCompliance
from app.models.faq import FAQArticle, OrganizationFAQArticle
from app.llm_trigger.service import invalidate_llm_trigger_cache

router = APIRouter()

POLICY_DOCS_FOLDER = "policy-docs"
SOP_DOCS_FOLDER = "sop-procedures"
LEGACY_FAQ_DOCS_FOLDER = "faq-docs"

# --- Schemas ---

class PolicyCreate(BaseModel):
    title: str
    category: str
    content: str

class PolicyUpdate(BaseModel):
    title: Optional[str] = None
    category: Optional[str] = None
    content: Optional[str] = None

class FAQCreate(BaseModel):
    question: str
    answer: str
    category: str

class FAQUpdate(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    category: Optional[str] = None


def _fallback_label(value: str | None, fallback: str) -> str:
    if value and value.strip():
        return value.strip()
    return fallback


def _extract_pdf_text(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF") from exc

    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(text)

    return "\n\n".join(pages).strip()


async def _get_org_slug(session: SessionDep, organization_id) -> str:
    result = await session.exec(select(Organization.slug).where(Organization.id == organization_id))
    slug = result.first()
    if not slug:
        raise HTTPException(status_code=404, detail="Organization not found")
    return slug


def _document_path(base_root: str, org_slug: str, folder_name: str, document_id: str) -> Path:
    return Path(base_root) / org_slug / folder_name / f"{document_id}.pdf"


async def _store_pdf_upload(
    session: SessionDep,
    current_user: CurrentUser,
    upload: UploadFile,
    base_root: str,
    folder_name: str,
    document_id: str,
) -> tuple[str, Path]:
    if not upload.filename or not upload.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await upload.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    org_slug = await _get_org_slug(session, current_user.organization_id)
    target_path = _document_path(base_root, org_slug, folder_name, document_id)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(file_bytes)
    return _extract_pdf_text(file_bytes), target_path


def _delete_document_file(base_root: str, org_slug: str, folder_name: str, document_id: str) -> None:
    target_path = _document_path(base_root, org_slug, folder_name, document_id)
    if target_path.exists():
        target_path.unlink()

# --- Endpoints ---


@router.get("/policies")
async def list_policies(session: SessionDep, current_user: CurrentUser):
    """List all company policies with usage metrics."""
    # Subquery for compliance count
    count_stmt = (
        select(PolicyCompliance.policy_id, func.count(PolicyCompliance.id).label("usage_count"))
        .group_by(PolicyCompliance.policy_id)
        .subquery()
    )

    result = await session.exec(
        select(CompanyPolicy, OrganizationPolicy, count_stmt.c.usage_count)
        .join(OrganizationPolicy, OrganizationPolicy.policy_id == CompanyPolicy.id)
        .outerjoin(count_stmt, count_stmt.c.policy_id == CompanyPolicy.id)
        .where(OrganizationPolicy.organization_id == current_user.organization_id)
        .order_by(CompanyPolicy.policy_title)
    )
    
    policies_data = result.all()

    return [
        {
            "id": str(p.id),
            "documentType": "policy",
            "title": p.policy_title,
            "category": p.policy_category,
            "content": p.policy_text,
            "preview": p.policy_text[:60] + "..." if len(p.policy_text) > 60 else p.policy_text,
            "lastUpdated": p.updated_at.strftime("%Y-%m-%d") if p.updated_at else "",
            "isActive": op.is_active,
            "usageCount": usage_count or 0,
        }
        for p, op, usage_count in policies_data
    ]


@router.post("/policies")
async def create_policy(session: SessionDep, current_user: CurrentUser, data: PolicyCreate):
    """Create a new policy and assign it to the current organization."""
    policy = CompanyPolicy(
        policy_title=data.title,
        policy_category=data.category,
        policy_text=data.content,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        updated_at=datetime.now(timezone.utc).replace(tzinfo=None)
    )
    session.add(policy)
    await session.flush() # Get policy.id

    org_link = OrganizationPolicy(
        organization_id=current_user.organization_id,
        policy_id=policy.id,
        is_active=True
    )
    session.add(org_link)
    await session.commit()
    await invalidate_llm_trigger_cache(session, org_filter=current_user.organization_id)
    return {"status": "success", "id": str(policy.id)}


@router.post("/policies/upload")
async def upload_policy(
    session: SessionDep,
    current_user: CurrentUser,
    title: str = Form(default=""),
    category: str = Form(default="Guidelines"),
    file: UploadFile = File(...),
):
    """Upload a PDF policy and store its extracted text in the knowledge base."""
    policy_id = uuid4()
    extracted_text, _ = await _store_pdf_upload(
        session,
        current_user,
        file,
        settings.POLICY_DOCS_ROOT,
        POLICY_DOCS_FOLDER,
        str(policy_id),
    )
    policy = CompanyPolicy(
        id=policy_id,
        policy_title=_fallback_label(title, Path(file.filename or "policy").stem.replace("_", " ").title()),
        policy_category=_fallback_label(category, "Guidelines"),
        policy_text=extracted_text or _fallback_label(title, "Uploaded policy document"),
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    session.add(policy)
    await session.flush()

    org_link = OrganizationPolicy(
        organization_id=current_user.organization_id,
        policy_id=policy.id,
        is_active=True,
    )
    session.add(org_link)
    await session.commit()
    await invalidate_llm_trigger_cache(session, org_filter=current_user.organization_id)
    return {"status": "success", "id": str(policy.id)}


@router.patch("/policies/{policy_id}")
async def update_policy(session: SessionDep, current_user: CurrentUser, policy_id: str, data: PolicyUpdate):
    """Update an existing policy."""
    statement = select(CompanyPolicy).where(CompanyPolicy.id == policy_id)
    result = await session.exec(statement)
    policy = result.first()
    if not policy:
        return {"status": "error", "message": "Policy not found"}

    if data.title:
        policy.policy_title = data.title
    if data.category:
        policy.policy_category = data.category
    if data.content:
        policy.policy_text = data.content
    policy.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    session.add(policy)
    await session.commit()
    return {"status": "success"}


@router.patch("/policies/{policy_id}/upload")
async def replace_policy_upload(
    session: SessionDep,
    current_user: CurrentUser,
    policy_id: str,
    title: str = Form(default=""),
    category: str = Form(default=""),
    file: UploadFile = File(...),
):
    """Replace an existing policy with a newer PDF version."""
    statement = select(CompanyPolicy).where(CompanyPolicy.id == policy_id)
    result = await session.exec(statement)
    policy = result.first()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")

    extracted_text, _ = await _store_pdf_upload(
        session,
        current_user,
        file,
        settings.POLICY_DOCS_ROOT,
        POLICY_DOCS_FOLDER,
        policy_id,
    )

    policy.policy_title = _fallback_label(title, policy.policy_title)
    policy.policy_category = _fallback_label(category, policy.policy_category)
    policy.policy_text = extracted_text or policy.policy_text
    policy.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    session.add(policy)
    await session.commit()
    await invalidate_llm_trigger_cache(session, org_filter=current_user.organization_id)
    return {"status": "success", "id": str(policy.id)}


@router.post("/policies/{policy_id}/toggle")
async def toggle_policy(session: SessionDep, current_user: CurrentUser, policy_id: str):
    """Toggle a policy's active status for the organization."""
    statement = select(OrganizationPolicy).where(
        OrganizationPolicy.organization_id == current_user.organization_id,
        OrganizationPolicy.policy_id == policy_id
    )
    result = await session.exec(statement)
    org_policy = result.first()
    if not org_policy:
        return {"status": "error", "message": "Assignment not found"}
    
    org_policy.is_active = not org_policy.is_active
    session.add(org_policy)
    await session.commit()
    await invalidate_llm_trigger_cache(session, org_filter=current_user.organization_id)
    return {"status": "success", "isActive": org_policy.is_active}


@router.get("/faqs")
async def list_faqs(session: SessionDep, current_user: CurrentUser):
    """List all FAQ articles."""
    result = await session.exec(
        select(FAQArticle, OrganizationFAQArticle)
        .join(OrganizationFAQArticle, OrganizationFAQArticle.article_id == FAQArticle.id)
        .where(OrganizationFAQArticle.organization_id == current_user.organization_id)
        .order_by(FAQArticle.question)
    )
    faqs_data = result.all()

    return [
        {
            "id": str(f.id),
            "documentType": "faq",
            "question": f.question,
            "answer": f.answer,
            "preview": f.answer[:60] + "..." if len(f.answer) > 60 else f.answer,
            "category": f.category,
            "isActive": of.is_active,
            "usageCount": 0, # FAQs usage tracking not implemented yet
        }
        for f, of in faqs_data
    ]


@router.post("/faqs")
async def create_faq(session: SessionDep, current_user: CurrentUser, data: FAQCreate):
    """Create a new FAQ and assign it to the current organization."""
    faq = FAQArticle(
        question=data.question,
        answer=data.answer,
        category=data.category,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        updated_at=datetime.now(timezone.utc).replace(tzinfo=None)
    )
    session.add(faq)
    await session.flush()

    org_link = OrganizationFAQArticle(
        organization_id=current_user.organization_id,
        article_id=faq.id,
        is_active=True
    )
    session.add(org_link)
    await session.commit()
    return {"status": "success", "id": str(faq.id)}


@router.post("/faqs/upload")
async def upload_faq(
    session: SessionDep,
    current_user: CurrentUser,
    question: str = Form(default=""),
    category: str = Form(default="Knowledge"),
    file: UploadFile = File(...),
):
    """Upload a PDF FAQ article and store the extracted answer text."""
    faq_id = uuid4()
    extracted_text, _ = await _store_pdf_upload(
        session,
        current_user,
        file,
        settings.KNOWLEDGE_DOCS_ROOT,
        SOP_DOCS_FOLDER,
        str(faq_id),
    )
    faq = FAQArticle(
        id=faq_id,
        question=_fallback_label(question, Path(file.filename or "faq").stem.replace("_", " ").title()),
        answer=extracted_text or _fallback_label(question, "Uploaded FAQ document"),
        category=_fallback_label(category, "Knowledge"),
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    session.add(faq)
    await session.flush()

    org_link = OrganizationFAQArticle(
        organization_id=current_user.organization_id,
        article_id=faq.id,
        is_active=True,
    )
    session.add(org_link)
    await session.commit()
    return {"status": "success", "id": str(faq.id)}


@router.patch("/faqs/{faq_id}")
async def update_faq(session: SessionDep, current_user: CurrentUser, faq_id: str, data: FAQUpdate):
    """Update an existing FAQ."""
    statement = select(FAQArticle).where(FAQArticle.id == faq_id)
    result = await session.exec(statement)
    faq = result.first()
    if not faq:
        return {"status": "error", "message": "FAQ not found"}

    if data.question:
        faq.question = data.question
    if data.answer:
        faq.answer = data.answer
    if data.category:
        faq.category = data.category
    faq.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    session.add(faq)
    await session.commit()
    return {"status": "success"}


@router.patch("/faqs/{faq_id}/upload")
async def replace_faq_upload(
    session: SessionDep,
    current_user: CurrentUser,
    faq_id: str,
    question: str = Form(default=""),
    category: str = Form(default=""),
    file: UploadFile = File(...),
):
    """Replace an existing FAQ with a newer PDF version."""
    statement = select(FAQArticle).where(FAQArticle.id == faq_id)
    result = await session.exec(statement)
    faq = result.first()
    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")

    extracted_text, _ = await _store_pdf_upload(
        session,
        current_user,
        file,
        settings.KNOWLEDGE_DOCS_ROOT,
        SOP_DOCS_FOLDER,
        faq_id,
    )

    faq.question = _fallback_label(question, faq.question)
    faq.answer = extracted_text or faq.answer
    faq.category = _fallback_label(category, faq.category)
    faq.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    session.add(faq)
    await session.commit()
    return {"status": "success", "id": str(faq.id)}


@router.post("/faqs/{faq_id}/toggle")
async def toggle_faq(session: SessionDep, current_user: CurrentUser, faq_id: str):
    """Toggle an FAQ's active status for the organization."""
    statement = select(OrganizationFAQArticle).where(
        OrganizationFAQArticle.organization_id == current_user.organization_id,
        OrganizationFAQArticle.article_id == faq_id
    )
    result = await session.exec(statement)
    org_faq = result.first()
    if not org_faq:
        return {"status": "error", "message": "Assignment not found"}
    
    org_faq.is_active = not org_faq.is_active
    session.add(org_faq)
    await session.commit()
    return {"status": "success", "isActive": org_faq.is_active}

@router.delete("/policies/{policy_id}")
async def delete_policy(
    session: SessionDep,
    current_user: CurrentUser,
    policy_id: str
):
    """
    Remove a policy from the organization's knowledge base.
    If it's the only reference, we could potentially delete the policy itself, 
    but for now, we just remove the association.
    """
    # Find the association
    stmt = select(OrganizationPolicy).where(
        OrganizationPolicy.organization_id == current_user.organization_id,
        OrganizationPolicy.policy_id == policy_id
    )
    res = await session.exec(stmt)
    org_policy = res.first()
    
    if not org_policy:
        return {"status": "error", "message": "Policy not found in your organization"}
        
    await session.delete(org_policy)
    
    # Check if any other organization uses this policy
    # If not, we can delete the global policy record too
    other_stmt = select(OrganizationPolicy).where(OrganizationPolicy.policy_id == policy_id)
    other_res = await session.exec(other_stmt)
    if not other_res.first():
        policy_stmt = select(CompanyPolicy).where(CompanyPolicy.id == policy_id)
        policy_res = await session.exec(policy_stmt)
        policy = policy_res.first()
        if policy:
            await session.delete(policy)
            org_slug = await _get_org_slug(session, current_user.organization_id)
            _delete_document_file(settings.POLICY_DOCS_ROOT, org_slug, POLICY_DOCS_FOLDER, policy_id)
            
    await session.commit()
    await invalidate_llm_trigger_cache(session, org_filter=current_user.organization_id)
    return {"status": "success", "message": "Policy deleted"}


@router.delete("/faqs/{faq_id}")
async def delete_faq(
    session: SessionDep,
    current_user: CurrentUser,
    faq_id: str
):
    """
    Remove an FAQ article from the organization's knowledge base.
    """
    stmt = select(OrganizationFAQArticle).where(
        OrganizationFAQArticle.organization_id == current_user.organization_id,
        OrganizationFAQArticle.article_id == faq_id
    )
    res = await session.exec(stmt)
    org_faq = res.first()
    
    if not org_faq:
        return {"status": "error", "message": "FAQ not found in your organization"}
        
    await session.delete(org_faq)
    
    # Clean up global FAQ if no one else uses it
    other_stmt = select(OrganizationFAQArticle).where(OrganizationFAQArticle.article_id == faq_id)
    other_res = await session.exec(other_stmt)
    if not other_res.first():
        faq_stmt = select(FAQArticle).where(FAQArticle.id == faq_id)
        faq_res = await session.exec(faq_stmt)
        faq = faq_res.first()
        if faq:
            await session.delete(faq)
            org_slug = await _get_org_slug(session, current_user.organization_id)
            _delete_document_file(settings.KNOWLEDGE_DOCS_ROOT, org_slug, SOP_DOCS_FOLDER, faq_id)
            _delete_document_file(settings.KNOWLEDGE_DOCS_ROOT, org_slug, LEGACY_FAQ_DOCS_FOLDER, faq_id)
            
    await session.commit()
    return {"status": "success", "message": "FAQ deleted"}
