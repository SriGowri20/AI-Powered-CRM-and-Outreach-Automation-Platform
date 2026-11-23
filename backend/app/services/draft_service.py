
"""
Service for generating campaign email drafts.

This service generates and stores drafts only.
It never sends or dispatches emails.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from backend.app.core.db import get_pool
from backend.app.services.eligibility import is_eligible_for_campaign

# Adjust this import to the actual module path in your project.
from backend.app.services.scoring_service import calculate_data_quality_score

from ai.gemini_client import GeminiClient
from ai.validators.fact_check import fact_check_draft


logger = logging.getLogger(__name__)

# draft_service.py is assumed to be:
# repo/backend/app/services/draft_service.py
REPO_ROOT = Path(__file__).resolve().parents[3]

PROMPT_DIR = REPO_ROOT / "ai" / "prompts"
BASE_PROMPT_PATH = PROMPT_DIR / "email_draft_v1.md"

TONE_PROMPT_PATHS = {
    "hr": PROMPT_DIR / "tone_hr.md",
    "tpo": PROMPT_DIR / "tone_tpo.md",
    "startup_founder": PROMPT_DIR / "tone_startup_founder.md",
}

PROMPT_VERSION = "email_draft_v1"
ALLOWED_TONES = set(TONE_PROMPT_PATHS)


class IneligibleLeadError(Exception):
    """Raised when a lead cannot be contacted for a campaign."""


class DraftNotFoundError(Exception):
    """Raised when a required campaign, opportunity, or contact is missing."""


class DraftService:
    def __init__(self, db=None, gemini_client=None):
        if db is None:
            db = get_pool()

        self.db = db
        self.gemini_client = gemini_client or GeminiClient()

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

        # Read database records before calling Gemini.
        opportunity = self._get_opportunity(opportunity_id)

        if not opportunity:
            raise DraftNotFoundError("Opportunity not found.")

        campaign, contact = self._get_campaign_and_contact(
            campaign_id,
            recipient_contact_method_id,
        )

        if not campaign:
            raise DraftNotFoundError("Campaign not found.")

        if not contact:
            raise DraftNotFoundError("Recipient contact method not found.")

        # Build scoring inputs from the actual database records.
        # Replace the mapping in this method with your real schema fields.
        score_result = self._calculate_opportunity_score(opportunity)

        if (
            score_result["band"] == "suppressed"
            or score_result["score"] < 80
            or not score_result["is_eligible"]
        ):
            raise IneligibleLeadError(
                "This lead does not meet the campaign eligibility score "
                "or has been suppressed."
            )

        # This function returns text, so compare the exact expected value.
        eligibility = is_eligible_for_campaign(opportunity)

        if eligibility != "Eligible":
            raise IneligibleLeadError(
                "This lead is not eligible for campaign outreach."
            )

        # Only send verified, flat facts to Gemini.
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
            draft_status=draft_status,
        )

    def _get_opportunity(self, opportunity_id: str):
        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM opportunities
                    WHERE id = %s
                    """,
                    (opportunity_id,),
                )
                return cur.fetchone()

    def _get_campaign_and_contact(
        self,
        campaign_id: str,
        recipient_contact_method_id: str,
    ):
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
                    SELECT *
                    FROM contact_methods
                    WHERE id = %s
                    """,
                    (recipient_contact_method_id,),
                )
                contact = cur.fetchone()

        return campaign, contact

    def _calculate_opportunity_score(
        self,
        opportunity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Map actual opportunity/database fields to the scoring function.

        IMPORTANT:
        The values below are placeholders until matched to your migrations
        and related tables. Do not assume these are actual DB column names.
        """

        return calculate_data_quality_score(
            source_type=opportunity["source_type"],
            has_contact_role=opportunity["has_contact_role"],
            verification_status=opportunity["verification_status"],
            published_days_ago=opportunity["published_days_ago"],
            has_description=opportunity["has_description"],
            has_salary=opportunity["has_salary"],
            has_location=opportunity["has_location"],
            is_suppressed=opportunity["is_suppressed"],
        )

    def _build_verified_facts(
        self,
        opportunity: Dict[str, Any],
        campaign: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return only verified facts.

        Replace these example source fields with fields confirmed in
        your database. Do not pass SELECT * rows or contact details.
        """

        facts = {
            "fact_role": opportunity.get("job_title"),
            "fact_location": opportunity.get("location"),
            "fact_source": opportunity.get("source"),
            "fact_campaign": campaign.get("name"),
        }

        # The supplied validator specifically looks for this key
        # when checking compensation claims.
        facts["salary_text"] = opportunity.get("salary_text") or ""

        # Remove unavailable values so the model cannot treat them
        # as confirmed facts.
        return {
            fact_id: value
            for fact_id, value in facts.items()
            if value is not None and value != ""
        }

    def _build_prompt(
        self,
        tone: str,
        verified_facts: Dict[str, Any],
    ) -> str:

        if not BASE_PROMPT_PATH.is_file():
            raise FileNotFoundError(
                f"Base prompt file not found: {BASE_PROMPT_PATH}"
            )

        tone_path = TONE_PROMPT_PATHS[tone]

        if not tone_path.is_file():
            raise FileNotFoundError(
                f"Tone prompt file not found: {tone_path}"
            )

        base_prompt = BASE_PROMPT_PATH.read_text(encoding="utf-8")
        tone_prompt = tone_path.read_text(encoding="utf-8")

        return (
            f"{base_prompt}\n\n"
            f"{tone_prompt}\n\n"
            "Verified facts (use only these facts):\n"
            f"{json.dumps(verified_facts, default=str, ensure_ascii=False)}"
        )

    def _generate_and_validate(
        self,
        prompt: str,
        verified_facts: Dict[str, Any],
    ):
        """
        Generate a draft and validate it.
        Retry once if validation fails.
        """

        validation_issues = []
        ai_output = {}

        for attempt in range(2):
            # GeminiClient.generate returns (output, usage).
            ai_output, _usage = self.gemini_client.generate(prompt)

            if not isinstance(ai_output, dict):
                validation_issues = [
                    "AI output must be a dictionary."
                ]
                continue

            subject = ai_output.get("subject", "")

            if not isinstance(subject, str) or len(subject) > 80:
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

        # Keep a non-NULL dictionary for the database JSONB column.
        if not isinstance(ai_output, dict):
            ai_output = {
                "subject": "",
                "body": "",
                "claims_used": [],
            }

        return ai_output, validation_issues

    def _save_draft(
        self,
        campaign_id: str,
        opportunity_id: str,
        recipient_contact_method_id: str,
        ai_output: Dict[str, Any],
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
                            ai_output,
                            status,
                            model,
                            prompt_version
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            campaign_id,
                            opportunity_id,
                            recipient_contact_method_id,
                            Jsonb(ai_output),
                            draft_status,
                            "gemini",
                            PROMPT_VERSION,
                        ),
                    )

                    saved = cur.fetchone()

        return {
            "draft_id": str(saved["id"]),
            "campaign_id": campaign_id,
            "status": draft_status,
            "ai_output": ai_output,
        }
