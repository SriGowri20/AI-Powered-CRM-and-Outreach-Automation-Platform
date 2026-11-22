from unittest.mock import MagicMock

from backend.app.services import draft_service as draft_service_module
from backend.app.services.draft_service import DraftService


def test_generate_draft_calls_mock_gemini_client(monkeypatch, tmp_path):
    # Mock Gemini's response.
    mock_ai_output = {
        "subject": "Opportunity for a Software Engineer",
        "body": "We have an opportunity that may interest you.",
        "claims_used": [
            {"fact_id": "job_title"}
        ],
    }

    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = mock_ai_output

    # The service expects a database dependency.
    # These methods are mocked so this test does not access a real DB.
    mock_db = MagicMock()

    service = DraftService(
        db=mock_db,
        gemini_client=mock_gemini,
    )

    # Supply sample records returned by the service's data-access methods.
    opportunity = {"id": "opportunity-123"}
    campaign = {"id": "campaign-123"}
    contact = {"id": "contact-123"}

    monkeypatch.setattr(
        service,
        "_get_opportunity",
        MagicMock(return_value=opportunity),
    )

    monkeypatch.setattr(
        service,
        "_get_campaign_and_contact",
        MagicMock(return_value=(campaign, contact)),
    )

    # Mock the eligibility check.
    monkeypatch.setattr(
        draft_service_module,
        "is_eligible_for_campaign",
        lambda opportunity: True,
    )

    # Mock the validator to accept the sample AI response.
    monkeypatch.setattr(
        draft_service_module,
        "fact_check_draft",
        lambda draft, verified_facts: (True, []),
    )

    # Provide a temporary prompt file.
    prompt_file = tmp_path / "email_draft_v1.md"
    prompt_file.write_text("Generate an email draft.", encoding="utf-8")

    monkeypatch.setattr(
        draft_service_module,
        "PROMPT_PATH",
        prompt_file,
    )

    # Mock saving the draft.
    saved_result = {
        "id": "draft-123",
        "status": "draft",
    }

    save_mock = MagicMock(return_value=saved_result)

    monkeypatch.setattr(
        service,
        "_save_draft",
        save_mock,
    )

    result = service.generate_draft(
        campaign_id="campaign-123",
        opportunity_id="opportunity-123",
        recipient_contact_method_id="contact-123",
        tone="hr",
    )

    # Verify Gemini was called.
    mock_gemini.generate.assert_called_once()

    # Verify the draft was saved.
    save_mock.assert_called_once()

    assert result == saved_result
