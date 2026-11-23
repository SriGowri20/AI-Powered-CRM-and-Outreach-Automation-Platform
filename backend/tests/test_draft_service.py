
import pytest

from backend.app.services import draft_service
from backend.app.services.draft_service import (
    DraftService,
    IneligibleLeadError,
)


@pytest.fixture
def opportunity():
    return {
        "id": "opp-123",
        "job_title": "Python Developer",
        "location": "Bengaluru",
        "source": "official_careers",

        # Inputs used by calculate_data_quality_score()
        "source_type": "official_careers",
        "has_contact_role": True,
        "verification_status": "deliverable",
        "published_days_ago": 1,
        "has_description": True,
        "has_salary": True,
        "has_location": True,
        "is_suppressed": False,

        "salary_text": "12 LPA",
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
def gemini_output():
    return {
        "subject": "Python Developer Opportunity",
        "body": (
            "We came across your Python Developer opportunity "
            "and wanted to share a relevant candidate."
        ),
        "claims_used": [
            {"fact_id": "fact_role"},
        ],
    }


@pytest.fixture
def service(
    monkeypatch,
    tmp_path,
    opportunity,
    campaign,
    contact,
):
    # Use temporary prompt files so tests do not depend on
    # the repository's prompt file contents.
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

    # Keep the actual scoring function active.
    # Only mock the separate eligibility text function.
    monkeypatch.setattr(
        draft_service,
        "is_eligible_for_campaign",
        lambda opportunity: "Eligible",
    )

    mock_gemini = type("MockGemini", (), {})()
    mock_gemini.generate = lambda prompt: (
        {
            "subject": "Test",
            "body": "Test body",
            "claims_used": [{"fact_id": "fact_role"}],
        },
        {},
    )

    instance = DraftService(
        db=object(),
        gemini_client=mock_gemini,
    )

    # Avoid database access in these service unit tests.
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

    # Capture the saved values instead of inserting into the DB.
    saved = {}

    def fake_save_draft(**kwargs):
        saved.update(kwargs)
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

    instance._test_saved = saved
    return instance


def test_generate_draft_calls_mock_gemini_client(
    service,
    gemini_output,
):
    class MockGemini:
        def __init__(self):
            self.called = False
            self.received_prompt = None

        def generate(self, prompt):
            self.called = True
            self.received_prompt = prompt
            return gemini_output, {"total_tokens": 100}

    mock_gemini = MockGemini()
    service.gemini_client = mock_gemini

    result = service.generate_draft(
        campaign_id="campaign-123",
        opportunity_id="opp-123",
        recipient_contact_method_id="contact-123",
        tone="hr",
    )

    assert mock_gemini.called is True
    assert "Verified facts" in mock_gemini.received_prompt
    assert result["status"] == "generated"
    assert result["ai_output"] == gemini_output


def test_generate_draft_rejects_score_below_80(
    service,
    monkeypatch,
):
    low_score_opportunity = {
        "id": "opp-low",
        "job_title": "Developer",
        "location": "Bengaluru",
        "source": "aggregator",
        "source_type": "aggregator",
        "has_contact_role": False,
        "verification_status": "unknown",
        "published_days_ago": 20,
        "has_description": False,
        "has_salary": False,
        "has_location": False,
        "is_suppressed": False,
        "salary_text": "",
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


def test_generate_draft_rejects_suppressed_lead(
    service,
    monkeypatch,
):
    suppressed_opportunity = {
        "id": "opp-suppressed",
        "job_title": "Python Developer",
        "location": "Bengaluru",
        "source": "official_careers",
        "source_type": "official_careers",
        "has_contact_role": True,
        "verification_status": "deliverable",
        "published_days_ago": 1,
        "has_description": True,
        "has_salary": True,
        "has_location": True,
        "is_suppressed": True,
        "salary_text": "12 LPA",
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


def test_validation_retry_then_needs_review(service):
    class MockGemini:
        def __init__(self):
            self.call_count = 0

        def generate(self, prompt):
            self.call_count += 1

            # Both outputs fail validation because body and claims
            # are empty. Each call returns (output, usage).
            return (
                {
                    "subject": "Test subject",
                    "body": "",
                    "claims_used": [],
                },
                {"total_tokens": 50},
            )

    mock_gemini = MockGemini()
    service.gemini_client = mock_gemini

    result = service.generate_draft(
        campaign_id="campaign-123",
        opportunity_id="opp-123",
        recipient_contact_method_id="contact-123",
        tone="hr",
    )

    assert mock_gemini.call_count == 2
    assert result["status"] == "needs_review"
    assert result["ai_output"] is not None
    assert result["ai_output"]["review_required"] is True
    assert result["ai_output"]["validation_issues"]


def test_invalid_tone_is_rejected(service):
    with pytest.raises(ValueError):
        service.generate_draft(
            campaign_id="campaign-123",
            opportunity_id="opp-123",
            recipient_contact_method_id="contact-123",
            tone="casual",
        )
