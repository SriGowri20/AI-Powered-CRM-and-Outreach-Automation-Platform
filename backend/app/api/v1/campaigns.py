from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from backend.app.core.security import require_roles
from backend.app.services.draft_service import DraftService

router = APIRouter(prefix="/campaigns", tags=["Campaigns & Drafts"])


class DraftRequest(BaseModel):
    opportunity_id: str
    recipient_contact_method_id: str
    tone: Optional[str] = "hr"


@router.post("/{id}/drafts")
def generate_draft(
    id: str,
    payload: DraftRequest,
    current_user: dict = Depends(
        require_roles(["admin", "campaign_manager"])
    ),
):
    try:
        service = DraftService()

        return service.generate_draft(
            campaign_id=id,
            opportunity_id=payload.opportunity_id,
            recipient_contact_method_id=payload.recipient_contact_method_id,
            tone=payload.tone,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
