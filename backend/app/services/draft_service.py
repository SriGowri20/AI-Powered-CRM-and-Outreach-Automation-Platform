
"""
Service for generating campaign email drafts.

This service generates and stores drafts only.
It never sends or dispatches emails.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from backend.app.core.db import get_pool
from backend.app.services.eligibility import is_eligible_for_campaign
from backend.app.services.scoring_service import calculate_data_quality_score

from ai.gemini_client import GeminiClient
from ai.validators.fact_check import fact_check_draft


logger = logging.getLogger(__name__)


# --------------------------------------------------
# Prompt configuration
# --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = REPO_ROOT / "ai" / "prompts"

PROMPT_PATH = PROMPT_DIR / "email_draft_v1.md"

TONE_PROMPT_PATHS = {
    "hr": PROMPT_DIR / "tone_hr.md",
    "tpo": PROMPT_DIR / "tone_tpo.md",
    "startup_founder": PROMPT_DIR / "tone_startup_founder.md",
}

PROMPT_VERSION = "email_draft_v1"
MODEL_NAME = "gemini"

ALLOWED_TONES = set(TONE_PROMPT_PATHS)


# --------------------------------------------------
# Exceptions
# --------------------------------------------------

class IneligibleLeadError(Exception):
    """Raised when a lead is not eligible for campaign outreach."""


class DraftNotFoundError(Exception):
    """Raised when a required campaign, opportunity, or contact is missing."""


# --------------------------------------------------
# Draft service
# --------------------------------------------------

class DraftService:

    def __init__(self, db=None, gemini_client=None):
        self.db = db if db is not None else get_pool()
        self.gemini_client = gemini_client or GeminiClient()

    # --------------------------------------------------
    # Main draft generation method
    # --------------------------------------------------

    def generate_draft(
        self,
        campaign_id: str,
        opportunity_id: str,
        recipient_contact_method_id: str,
        tone: str = "hr",
    ) -> Dict[str, Any]:

        if tone not in ALLOWED_TONES:
            raise ValueError(
                "Invalid tone. Allowed values: hr, tpo, startup_founder."
            )

        opportunity = self._get_opportunity(opportunity_id)

        if not opportunity:
            raise DraftNotFoundError("Opportunity not found.")

        campaign, contact = self._get_campaign_and_contact(
            campaign_id=campaign_id,
            opportunity_id=opportunity_id,
            recipient_contact_method_id=recipient_contact_method_id,
        )

        if not campaign:
            raise DraftNotFoundError("Campaign not found.")

        if not contact:
            raise DraftNotFoundError(
                "Recipient contact method not found or not linked "
                "to this opportunity."
            )

        score_result = self._calculate_opportunity_score(
            opportunity,
            contact,
        )

        if (
            score_result["band"] == "suppressed"
            or score_result["score"] < 80
            or not score_result["is_eligible"]
        ):
            raise IneligibleLeadError(
                "This lead does not meet the campaign eligibility score "
                "or has been suppressed."
            )

        # Eligibility returns text, not a boolean.
        eligibility = is_eligible_for_campaign(opportunity)

        if eligibility != "Eligible":
            raise IneligibleLeadError(
                "This lead is not eligible for campaign outreach."
            )

        verified_facts = self._build_verified_facts(
            opportunity=opportunity,
            campaign=campaign,
        )

        prompt = self._build_prompt(
            tone=tone,
            verified_facts=verified_facts,
        )

        ai_output, validation_issues = self._generate_and_validate(
            prompt=prompt,
            verified_facts=verified_facts,
        )

        if validation_issues:
            draft_status = "needs_review"
            ai_output["validation_issues"] = validation_issues
            ai_output["review_required"] = True
        else:
            draft_status = "generated"

        return self._save_draft(
            campaign_id=campaign_id,
            opportunity_id=opportunity_id,
            recipient_contact_method_id=recipient_contact_method_id,
            ai_output=ai_output,
            input_refs=verified_facts,
            draft_status=draft_status,
        )

    # --------------------------------------------------
    # Database reads
    # --------------------------------------------------

    def _get_opportunity(
        self,
        opportunity_id: str,
    ) -> Dict[str, Any]:

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        o.*,
                        s.name AS source_name,
                        s.status AS source_status,
                        s.is_enabled AS source_is_enabled,
                        s.permitted_use_note
                    FROM opportunities o
                    JOIN sources s
                        ON s.id = o.source_id
                    WHERE o.id = %s
                    """,
                    (opportunity_id,),
                )

                return cur.fetchone()

    def _get_campaign_and_contact(
        self,
        campaign_id: str,
        opportunity_id: str,
        recipient_contact_method_id: str,
    ) -> Tuple[Any, Any]:

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:

                cur.execute(
                    """
                    SELECT *
                    FROM campaigns
                    WHERE id = %s
                    """,
                    (campaign_id,),
                )

                campaign = cur.fetchone()

                cur.execute(
                    """
                    SELECT
                        cm.id,
                        cm.contact_id,
                        cm.kind,
                        cm.value,
                        cm.provenance,
                        cm.verified_status,
                        c.name AS contact_name,
                        c.role_title
                    FROM contact_methods cm
                    JOIN contacts c
                        ON c.id = cm.contact_id
                    JOIN opportunity_contacts oc
                        ON oc.contact_id = c.id
                    WHERE cm.id = %s
                      AND oc.opportunity_id = %s
                    """,
                    (
                        recipient_contact_method_id,
                        opportunity_id,
                    ),
                )

                contact = cur.fetchone()

        return campaign, contact

    # --------------------------------------------------
    # Eligibility scoring
    # --------------------------------------------------

    def _calculate_opportunity_score(
        self,
        opportunity: Dict[str, Any],
        contact: Dict[str, Any],
    ) -> Dict[str, Any]:

        published_at = opportunity.get("published_at")

        if published_at:
            if published_at.tzinfo is None:
                published_at = published_at.replace(
                    tzinfo=timezone.utc
                )

            published_days_ago = max(
                0,
                (
                    datetime.now(timezone.utc) - published_at
                ).days,
            )
        else:
            published_days_ago = 9999

        # source_id contains the source classification used by scoring.
        source_type = opportunity.get("source_id", "unknown")

        # Do not treat disabled or quarantined sources as trusted.
        if (
            not opportunity.get("source_is_enabled", False)
            or opportunity.get("source_status") != "healthy"
        ):
            source_type = "unknown"

        return calculate_data_quality_score(
            source_type=source_type,
            has_contact_role=bool(contact.get("role_title")),
            verification_status=contact.get(
                "verified_status",
                "unknown",
            ),
            published_days_ago=published_days_ago,
            has_description=bool(opportunity.get("description")),
            has_salary=bool(opportunity.get("salary_text")),
            has_location=bool(opportunity.get("location")),
            is_suppressed=bool(
                opportunity.get("is_suppressed", False)
            ),
        )

    # --------------------------------------------------
    # Verified facts
    # --------------------------------------------------

    def _build_verified_facts(
        self,
        opportunity: Dict[str, Any],
        campaign: Dict[str, Any],
    ) -> Dict[str, Any]:

        facts = {
            "fact_role": opportunity.get("title"),
            "fact_location": opportunity.get("location"),
            "fact_source": opportunity.get("source_id"),
            "fact_campaign": campaign.get("name"),
            "salary_text": opportunity.get("salary_text") or "",
        }

        return {
            fact_id: value
            for fact_id, value in facts.items()
            if value is not None and value != ""
        }

    # --------------------------------------------------
    # Prompt building
    # --------------------------------------------------

    def _build_prompt(
        self,
        tone: str,
        verified_facts: Dict[str, Any],
    ) -> str:

        if not PROMPT_PATH.is_file():
            raise FileNotFoundError(
                f"Base prompt file not found: {PROMPT_PATH}"
            )

        tone_path = TONE_PROMPT_PATHS[tone]

        if not tone_path.is_file():
            raise FileNotFoundError(
                f"Tone prompt file not found: {tone_path}"
            )

        base_prompt = PROMPT_PATH.read_text(encoding="utf-8")
        tone_prompt = tone_path.read_text(encoding="utf-8")

        facts_json = json.dumps(
            verified_facts,
            default=str,
            ensure_ascii=False,
        )

        return (
            f"{base_prompt}\n\n"
            f"{tone_prompt}\n\n"
            "Verified facts (use only these facts):\n"
            f"{facts_json}"
        )

    # --------------------------------------------------
    # Gemini generation and validation
    # --------------------------------------------------

    def _generate_and_validate(
        self,
        prompt: str,
        verified_facts: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], list]:

        validation_issues = []
        ai_output = {}

        # Initial attempt + one retry.
        for attempt in range(2):

            # GeminiClient.generate returns (output, usage).
            ai_output, _usage = self.gemini_client.generate(prompt)

            if not isinstance(ai_output, dict):
                validation_issues = [
                    "AI output must be a dictionary."
                ]
                ai_output = {}

                logger.warning(
                    "Gemini returned invalid output on attempt %s.",
                    attempt + 1,
                )
                continue

            subject = ai_output.get("subject", "")

            if not isinstance(subject, str):
                is_valid = False
                issues = [
                    "Email subject must be a string."
                ]

            elif len(subject) > 80:
                is_valid = False
                issues = [
                    "Email subject must be 80 characters or fewer."
                ]

            else:
                is_valid, issues = fact_check_draft(
                    draft=ai_output,
                    verified_facts=verified_facts,
                )

            if is_valid:
                return ai_output, []

            validation_issues = issues

            logger.warning(
                "Draft validation failed on attempt %s: %s",
                attempt + 1,
                issues,
            )

        # ai_output must never be None because the DB column is NOT NULL.
        if not isinstance(ai_output, dict):
            ai_output = {
                "subject": "",
                "body": "",
                "claims_used": [],
            }

        return ai_output, validation_issues

    # --------------------------------------------------
    # Save generated draft
    # --------------------------------------------------

    def _save_draft(
        self,
        campaign_id: str,
        opportunity_id: str,
        recipient_contact_method_id: str,
        ai_output: Dict[str, Any],
        input_refs: Dict[str, Any],
        draft_status: str,
    ) -> Dict[str, Any]:

        with self.db.connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cur:

                    cur.execute(
                        """
                        INSERT INTO campaign_drafts (
                            campaign_id,
                            opportunity_id,
                            recipient_contact_method_id,
                            model,
                            prompt_version,
                            input_refs,
                            ai_output,
                            status
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            campaign_id,
                            opportunity_id,
                            recipient_contact_method_id,
                            MODEL_NAME,
                            PROMPT_VERSION,
                            Jsonb(input_refs),
                            Jsonb(ai_output),
                            draft_status,
                        ),
                    )

                    saved = cur.fetchone()

        return {
            "draft_id": str(saved["id"]),
            "campaign_id": campaign_id,
            "status": draft_status,
            "ai_output": ai_output,
        }
