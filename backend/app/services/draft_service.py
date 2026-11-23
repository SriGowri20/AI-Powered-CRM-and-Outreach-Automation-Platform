"""
Service for generating campaign email drafts.

Uses direct SQL with psycopg3.
Does not send or dispatch emails.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ai.gemini_client import GeminiClient
from ai.validators.fact_check import fact_check_draft

from backend.app.services.eligibility import is_eligible_for_campaign

logger = logging.getLogger(__name__)


# draft_service.py is:
# repo/backend/app/services/draft_service.py
#
# parents[3] -> repo
REPO_ROOT = Path(__file__).resolve().parents[3]

PROMPT_PATH = (
    REPO_ROOT
    / "ai"
    / "prompts"
    / "email_draft_v1.md"
)

TONE_PROMPT_PATHS = {
    "hr": REPO_ROOT / "ai" / "prompts" / "tone_hr.md",
    "tpo": REPO_ROOT / "ai" / "prompts" / "tone_tpo.md",
    "startup_founder": (
        REPO_ROOT
        / "ai"
        / "prompts"
        / "tone_startup_founder.md"
    ),
}

ALLOWED_TONES = {
    "hr",
    "tpo",
    "startup_founder",
}

PROMPT_VERSION = "email_draft_v1"
MODEL_NAME = "gemini"


class IneligibleLeadError(Exception):
    """Opportunity is not eligible for campaign outreach."""


class DraftService:

    def __init__(
        self,
        db,
        gemini_client,
        scoring_service,
        suppression_service,
    ):
        """
        Dependencies are injected so the service can be tested.

        db:
            Expected to be a psycopg3 ConnectionPool.

        gemini_client:
            Existing GeminiClient instance.

        scoring_service:
            Existing project scoring service.

        suppression_service:
            Existing project suppression service.
        """
        self.db = db
        self.gemini_client = gemini_client
        self.scoring_service = scoring_service
        self.suppression_service = suppression_service

    def generate_draft(
        self,
        campaign_id: str,
        opportunity_id: str,
        recipient_contact_method_id: str,
        tone: str = "hr",
    ) -> Dict[str, Any]:

        # ---------------------------------------------------------
        # 1. Validate tone
        # ---------------------------------------------------------
        if tone not in ALLOWED_TONES:
            raise ValueError(
                "tone must be one of: hr, tpo, startup_founder"
            )

        # ---------------------------------------------------------
        # 2. Get opportunity
        # ---------------------------------------------------------
        opportunity = self._get_opportunity(opportunity_id)

        if not opportunity:
            raise ValueError("Opportunity not found.")

        # ---------------------------------------------------------
        # 3. Get score from scoring service
        # ---------------------------------------------------------
        score = self.scoring_service.get_score(opportunity_id)

        # Do not read score from opportunities table.
        if score < 80:
            raise IneligibleLeadError(
                "This lead is not eligible for campaign outreach."
            )

        # ---------------------------------------------------------
        # 4. Check suppression
        # ---------------------------------------------------------
        suppressed = self.suppression_service.is_suppressed(
            opportunity_id
        )

        if suppressed:
            raise IneligibleLeadError(
                "This lead is suppressed and cannot be contacted."
            )

        # ---------------------------------------------------------
        # 5. Check actual campaign eligibility
        # ---------------------------------------------------------
        eligibility = is_eligible_for_campaign(opportunity)

        # IMPORTANT:
        # This function returns text such as "Eligible", "Held", etc.
        if eligibility != "Eligible":
            raise IneligibleLeadError(
                "This lead is not eligible for campaign outreach."
            )

        # ---------------------------------------------------------
        # 6. Get campaign and contact method
        # ---------------------------------------------------------
        campaign, contact_method = self._get_campaign_and_contact(
            campaign_id=campaign_id,
            contact_method_id=recipient_contact_method_id,
        )

        if not campaign:
            raise ValueError("Campaign not found.")

        if not contact_method:
            raise ValueError("Contact method not found.")

        # ---------------------------------------------------------
        # 7. Build FLAT verified facts
        #
        # Do NOT send SELECT * rows or contact details to Gemini.
        # ---------------------------------------------------------
        verified_facts = self._build_verified_facts(
            campaign=campaign,
            opportunity=opportunity,
        )

        # ---------------------------------------------------------
        # 8. Load base + tone prompts
        # ---------------------------------------------------------
        prompt_template = PROMPT_PATH.read_text(
            encoding="utf-8"
        )

        tone_prompt = TONE_PROMPT_PATHS[tone].read_text(
            encoding="utf-8"
        )

        prompt = (
            prompt_template
            + "\n\n"
            + tone_prompt
            + "\n\n"
            + "Verified facts:\n"
            + json.dumps(
                verified_facts,
                ensure_ascii=False,
            )
        )

        # ---------------------------------------------------------
        # 9. Generate + validate.
        #    Maximum 2 attempts.
        # ---------------------------------------------------------
        ai_output = None
        validation_issues = []

        for attempt in range(2):

            # GeminiClient.generate returns:
            # (output, usage)
            ai_output, _ = self.gemini_client.generate(prompt)

            if not isinstance(ai_output, dict):
                validation_issues = [
                    "AI output must be a dictionary."
                ]

                logger.warning(
                    "AI output validation failed on attempt %s",
                    attempt + 1,
                )
                continue

            # Subject validation
            subject = ai_output.get("subject", "")

            if not isinstance(subject, str):
                validation_issues = [
                    "Email subject must be a string."
                ]

                logger.warning(
                    "Draft validation failed on attempt %s: %s",
                    attempt + 1,
                    validation_issues,
                )
                continue

            if len(subject) > 80:
                validation_issues = [
                    "Email subject must be 80 characters or fewer."
                ]

                logger.warning(
                    "Draft validation failed on attempt %s: %s",
                    attempt + 1,
                    validation_issues,
                )
                continue

            # Fact validation
            is_valid, issues = fact_check_draft(
                draft=ai_output,
                verified_facts=verified_facts,
            )

            if is_valid:
                validation_issues = []
                break

            validation_issues = issues

            logger.warning(
                "Draft validation failed on attempt %s: %s",
                attempt + 1,
                issues,
            )

        # ---------------------------------------------------------
        # 10. Determine final status
        # ---------------------------------------------------------
        if validation_issues:

            draft_status = "needs_review"

            # ai_output is NOT NULL in database.
            if not isinstance(ai_output, dict):
                ai_output = {}

            ai_output["validation_issues"] = validation_issues
            ai_output["review_required"] = True

        else:
            draft_status = "generated"

        # ---------------------------------------------------------
        # 11. Persist
        # ---------------------------------------------------------
        return self._save_draft(
            campaign_id=campaign_id,
            opportunity_id=opportunity_id,
            contact_method_id=recipient_contact_method_id,
            ai_output=ai_output,
            draft_status=draft_status,
        )

    def _get_opportunity(
        self,
        opportunity_id: str,
    ):
        """
        Fetch only fields required for eligibility/facts.

        Do not use SELECT *.
        """

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:

                cur.execute(
                    """
                    SELECT
                        id,
                        job_title,
                        location,
                        salary_text,
                        source
                    FROM opportunities
                    WHERE id = %s
                    """,
                    (opportunity_id,),
                )

                return cur.fetchone()

    def _get_campaign_and_contact(
        self,
        campaign_id: str,
        contact_method_id: str,
    ):
        """
        Fetch campaign and contact method.

        Contact details are NOT included in verified_facts
        sent to Gemini.
        """

        with self.db.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:

                cur.execute(
                    """
                    SELECT
                        id,
                        name,
                        description
                    FROM campaigns
                    WHERE id = %s
                    """,
                    (campaign_id,),
                )

                campaign = cur.fetchone()

                cur.execute(
                    """
                    SELECT
                        id,
                        channel
                    FROM contact_methods
                    WHERE id = %s
                    """,
                    (contact_method_id,),
                )

                contact_method = cur.fetchone()

                return campaign, contact_method

    def _build_verified_facts(
        self,
        campaign: Dict[str, Any],
        opportunity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a flat dictionary keyed by fact_id.

        IMPORTANT:
        These are the only facts sent to Gemini.
        """

        return {
            "fact_role": opportunity.get("job_title"),
            "fact_location": opportunity.get("location"),
            "fact_salary": opportunity.get("salary_text"),
            "fact_source": opportunity.get("source"),
            "fact_campaign_name": campaign.get("name"),
            "fact_campaign_description": campaign.get(
                "description"
            ),
        }

    def _save_draft(
        self,
        campaign_id: str,
        opportunity_id: str,
        contact_method_id: str,
        ai_output: Dict[str, Any],
        draft_status: str,
    ):
        """Persist generated draft using raw SQL."""

        if ai_output is None:
            ai_output = {}

        with self.db.connection() as conn:
            with conn.transaction():

                with conn.cursor(row_factory=dict_row) as cur:

                    cur.execute(
                        """
                        INSERT INTO campaign_drafts (
                            campaign_id,
                            opportunity_id,
                            contact_method_id,
                            ai_output,
                            status,
                            model,
                            prompt_version
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s
                        )
                        RETURNING id
                        """,
                        (
                            campaign_id,
                            opportunity_id,
                            contact_method_id,
                            Jsonb(ai_output),
                            draft_status,
                            MODEL_NAME,
                            PROMPT_VERSION,
                        ),
                    )

                    saved = cur.fetchone()

        return {
            "draft_id": str(saved["id"]),
            "campaign_id": campaign_id,
            "opportunity_id": opportunity_id,
            "contact_method_id": contact_method_id,
            "status": draft_status,
            "ai_output": ai_output,
        }
