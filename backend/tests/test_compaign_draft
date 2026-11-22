import pytest
from fastapi.testclient import TestClient

from backend.app.main import app  # Adjust if your FastAPI app is defined elsewhere
from backend.app.api.v1 import campaigns as campaigns_api
from backend.app.services.draft_service import IneligibleLeadError

client = TestClient(app)


def test_non_eligible_lead_returns_409(monkeypatch):
    class FakeDraftService:
        def __init__(self, **kwargs):
            pass

        def generate_draft(self, **kwargs):
            raise IneligibleLeadError(
                "This lead is not eligible for campaign outreach."
            )

    # Replace the real service with our fake service.
    monkeypatch.setattr(
        campaigns_api,
        "DraftService",
        FakeDraftService,
    )

    response = client.post(
        "/api/v1/campaigns/campaign-123/drafts",
        json={
            "opportunity_id": "opportunity-123",
            "recipient_contact_method_id": "contact-123",
            "tone": "hr",
        },
    )

    assert response.status_code == 409
    assert "not eligible" in response.json()["detail"].lower()
