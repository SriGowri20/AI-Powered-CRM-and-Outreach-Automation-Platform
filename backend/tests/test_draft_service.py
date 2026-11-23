
import pytest

from backend.app.services import draft_service
from backend.app.services.draft_service import (
    DraftService,
    IneligibleLeadError,
    DraftNotFoundError,
)


@pytest.fixture
def opportunity():
    return {
        "id": "opp-123",
        "job_title": "Python Developer",
        "location": "Bengaluru",
        "source": "official_careers",
        "salary_text": "12 LPA",

        # Scoring service inputs.
        # These must eventually match your real DB mapping.
        "source_type": "official_careers",
        "has_contact_role": True,
        "verification_status": "deliverable",
        "published_days_ago": 1,
        "has_description": True,
        "has_salary": True,
        "has_location": True,
        "is_suppressed": False,
    }


@pytest.fixture
def campaign():
    return {
        "id": "campaign-123",
        "name": "Developer Outreach",
    }


@pytest.fixture
def contact():
    return {
        "id": "contact-123",
    }


@pytest.fixture
def prompt_files(tmp_path, monkeypatch):
    base_prompt = tmp_path / "email_draft_v1.md"
    tone_prompt = tmp_path / "tone_hr.md"

    base_prompt.write_text(
        "Generate an email using only verified facts.",
        encoding="utf-8",
    )

    tone_prompt.write_text(
        "Use a professional HR tone.",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        draft_service,
        "PROMPT_PATH",
        base_prompt,
    )

    monkeypatch.setattr(
        draft_service,
        "TONE_PROMPT_PATHS",
        {"hr": tone_prompt},
    )


@pytest.fixture
def service(
    monkeypatch,
    opportunity,
    campaign,
    contact,
    prompt_files,
):
    # The real scoring function remains active.
    # Only the separate eligibility text function is mocked.
    monkeypatch.setattr(
        draft_service,
        "is_eligible_for_campaign",
        lambda opportunity: "Eligible",
    )

    mock_gemini = MockGemini()

    instance = DraftService(
        db=object(),
        gemini_client=mock_gemini,
    )

    # Mock database reads so these tests do not need PostgreSQL.
    monkeypatch.setattr(
        instance,
        "_get_opportunity",
        lambda opportunity_id: opportunity,
    )

    monkeypatch.setattr(
        instance,
        "_get_campaign_and_contact",
        lambda campaign_id, contact_id: (campaign, contact),
    )

    # Capture the data that would be saved to the database.
    saved_draft = {}

    def fake_save_draft(**kwargs):
        saved_draft.update(kwargs)

        return {
            "draft_id": "draft-123",
            "campaign_id": kwargs["campaign_id"],
            "status": kwargs["draft_status"],
            "ai_output": kwargs["ai_output"],
        }

    monkeypatch.setattr(
        instance,
        "_save_draft",
        fake_save_draft,
    )

    instance.saved_draft = saved_draft

    return instance


class MockGemini:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        self.call_count += 1

        if self.responses:
            return self.responses.pop(0)

        return (
            {
                "subject": "Python Developer Opportunity",
                "body": (
                    "We found a Python Developer opportunity "
                    "that may interest you."
                ),
                "claims_used": [
                    {"fact_id": "fact_role"},
                ],
            },
            {"total_tokens": 100},
        )


def test_generate_draft_calls_mock_gemini_client(service):
    result = service.generate_draft(
        campaign_id="campaign-123",
        opportunity_id="opp-123",
        recipient_contact_method_id="contact-123",
        tone="hr",
    )

    assert service.gemini_client.call_count == 1

    assert "Verified facts" in service.gemini_client.prompts[0]

    assert result["status"] == "generated"
    assert result["draft_id"] == "draft-123"

    assert result["ai_output"]["subject"]
    assert result["ai_output"]["body"]

    assert service.saved_draft["draft_status"] == "generated"
    assert service.saved_draft["ai_output"] is not None


def test_generate_draft_rejects_score_below_80(
    service,
    monkeypatch,
):
    low_score_opportunity = {
        "id": "opp-low",
        "job_title": "Developer",
        "location": "Bengaluru",
        "source": "aggregator",
        "salary_text": "",

        "source_type": "aggregator",
        "has_contact_role": False,
        "verification_status": "unknown",
        "published_days_ago": 20,
        "has_description": False,
        "has_salary": False,
        "has_location": False,
        "is_suppressed": False,
    }

    monkeypatch.setattr(
        service,
        "_get_opportunity",
        lambda opportunity_id: low_score_opportunity,
    )

    with pytest.raises(IneligibleLeadError):
        service.generate_draft(
            campaign_id="campaign-123",
            opportunity_id="opp-low",
            recipient_contact_method_id="contact-123",
            tone="hr",
        )

    # Gemini must not run for an ineligible lead.
    assert service.gemini_client.call_count == 0


def test_generate_draft_rejects_suppressed_lead(
    service,
    monkeypatch,
):
    suppressed_opportunity = {
        "id": "opp-suppressed",
        "job_title": "Python Developer",
        "location": "Bengaluru",
        "source": "official_careers",
        "salary_text": "12 LPA",

        "source_type": "official_careers",
        "has_contact_role": True,
        "verification_status": "deliverable",
        "published_days_ago": 1,
        "has_description": True,
        "has_salary": True,
        "has_location": True,
        "is_suppressed": True,
    }

    monkeypatch.setattr(
        service,
        "_get_opportunity",
        lambda opportunity_id: suppressed_opportunity,
    )

    with pytest.raises(IneligibleLeadError):
        service.generate_draft(
            campaign_id="campaign-123",
            opportunity_id="opp-suppressed",
            recipient_contact_method_id="contact-123",
            tone="hr",
        )

    assert service.gemini_client.call_count == 0


def test_validation_retry_then_needs_review(service):
    invalid_output = {
        "subject": "Test subject",
        "body": "",
        "claims_used": [],
    }

    service.gemini_client = MockGemini(
        responses=[
            (invalid_output, {"total_tokens": 50}),
            (invalid_output, {"total_tokens": 60}),
        ]
    )

    result = service.generate_draft(
        campaign_id="campaign-123",
        opportunity_id="opp-123",
        recipient_contact_method_id="contact-123",
        tone="hr",
    )

    assert service.gemini_client.call_count == 2

    assert result["status"] == "needs_review"

    assert result["ai_output"] is not None
    assert result["ai_output"]["review_required"] is True
    assert result["ai_output"]["validation_issues"]

    assert service.saved_draft["draft_status"] == "needs_review"
    assert service.saved_draft["ai_output"] is not None


def test_invalid_tone_is_rejected(service):
    with pytest.raises(ValueError, match="Invalid tone"):
        service.generate_draft(
            campaign_id="campaign-123",
            opportunity_id="opp-123",
            recipient_contact_method_id="contact-123",
            tone="casual",
        )

    assert service.gemini_client.call_count == 0


def test_missing_opportunity_raises_not_found(service, monkeypatch):
    monkeypatch.setattr(
        service,
        "_get_opportunity",
        lambda opportunity_id: None,
    )

    with pytest.raises(DraftNotFoundError, match="Opportunity not found"):
        service.generate_draft(
            campaign_id="campaign-123",
            opportunity_id="missing",
            recipient_contact_method_id="contact-123",
            tone="hr",
        )
