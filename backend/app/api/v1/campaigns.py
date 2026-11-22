
"""
Campaign Management and AI Draft Generation Router.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.core.security import require_roles
from backend.app.services.draft_service import (
    DraftService,
    IneligibleLeadError,
)

# Import GeminiClient using your project's actual import path.
from ai.gemini_client import GeminiClient


router = APIRouter(
    prefix="/campaigns",
    tags=["Campaigns & Drafts"],
)


class DraftRequest(BaseModel):
    opportunity_id: str
    recipient_contact_method_id: str
    tone: Optional[str] = Field(default="hr")


@router.get("")
def list_campaigns(
    current_user: dict = Depends(
        require_roles(["admin", "campaign_manager"])
    ),
):
    """Lists available campaigns."""

    # Retain your existing GET /campaigns implementation here.
    # Do not remove its existing database or response logic.

    return [
        {
            "id": "d0000001-0000-0000-0000-000000000001",
            "name": "Q4 2026 Fresher Talent Outreach",
            "purpose": (
                "Connecting verified technical freshers "
                "with tech startups and scaleups"
            ),
            "status": "active",
            "daily_limit": 50,
        }
    ]


@router.post(
    "/{id}/drafts",
    status_code=status.HTTP_201_CREATED,
)
def generate_draft(
    id: str,
    payload: DraftRequest,
    current_user: dict = Depends(
        require_roles(["admin", "campaign_manager"])
    ),
):
    """
    Generate an AI campaign draft using verified opportunity facts.

    This endpoint only generates and stores drafts.
    It never dispatches messages.
    """

    try:
        service = DraftService(
            gemini_client=GeminiClient(),
        )

        result = service.generate_draft(
            campaign_id=id,
            opportunity_id=payload.opportunity_id,
            recipient_contact_method_id=(
                payload.recipient_contact_method_id
            ),
            tone=payload.tone or "hr",
        )

        return result

    except IneligibleLeadError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
