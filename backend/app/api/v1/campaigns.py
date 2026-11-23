
"""
Campaign Management and AI Draft Generation Router.
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.app.core.db import get_pool
from backend.app.core.security import require_roles

from backend.app.services.draft_service import (
    DraftService,
    IneligibleLeadError,
    DraftNotFoundError,
)

from ai.gemini_client import GeminiClient


router = APIRouter(
    prefix="/campaigns",
    tags=["Campaigns & Drafts"],
)


class DraftRequest(BaseModel):
    opportunity_id: str
    recipient_contact_method_id: str
    tone: Literal["hr", "tpo", "startup_founder"] = "hr"


@router.get("")
def list_campaigns(
    current_user: dict = Depends(
        require_roles(["admin", "campaign_manager"])
    ),
):
    """
    Lists available campaigns.

    IMPORTANT:
    Preserve your original GET /campaigns database query
    and response logic here. Do not replace it with sample data.
    """

    # Keep your existing GET /campaigns implementation here.
    pass


@router.post(
    "/{id}/drafts",
    status_code=status.HTTP_200_OK,
)
def generate_draft(
    id: str,
    payload: DraftRequest,
    current_user: dict = Depends(
        require_roles(["admin", "campaign_manager"])
    ),
):
    """
    Generate and save an AI campaign draft.

    This endpoint does not send or dispatch messages.
    """

    try:
        service = DraftService(
            db=get_pool(),
            gemini_client=GeminiClient(),
        )

        result = service.generate_draft(
            campaign_id=id,
            opportunity_id=payload.opportunity_id,
            recipient_contact_method_id=(
                payload.recipient_contact_method_id
            ),
            tone=payload.tone,
        )

        return result

    except IneligibleLeadError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except DraftNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
