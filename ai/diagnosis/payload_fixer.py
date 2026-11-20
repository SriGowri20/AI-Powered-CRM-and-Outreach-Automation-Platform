import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ai.config import get_ai_config
from ai.gemini_client import GeminiClient
from ai.diagnosis.error_diagnoser import redact_sensitive_data

logger = logging.getLogger(__name__)


class PayloadChange(BaseModel):
    """Describe one proposed change to a JSON payload."""

    model_config = ConfigDict(extra="forbid")

    path: str
    issue: str
    new_value: Any = None


class PayloadFixOutput(BaseModel):
    """Structured output for payload correction suggestions."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    corrected_payload: Optional[Dict[str, Any]] = None
    changes: List[PayloadChange] = Field(default_factory=list)
    needs_human_input: List[str] = Field(default_factory=list)
    retry_parameters: Optional[Dict[str, Any]] = None

def _fallback_payload_fix() -> Dict[str, Any]:
    """Return a safe response if payload analysis fails."""
    return PayloadFixOutput(
        summary=(
            "Payload analysis is unavailable. "
            "Manual review is required."
        ),
        corrected_payload=None,
        changes=[],
        needs_human_input=[
            "Review the original payload and error message."
        ],
        retry_parameters=None,
    ).model_dump()

def _mock_payload_fix(payload: Dict[str, Any], error_message: str) -> Dict[str, Any]:
    """Return a safe offline response for mock-mode testing."""
    return PayloadFixOutput(
        summary=(
            "Mock mode: payload analysis requires manual review."
        ),
        corrected_payload=None,
        changes=[],
        needs_human_input=[
            "Review the failing payload and error message."
        ],
        retry_parameters=None,
    ).model_dump()

def suggest_payload_fix(
    payload: Dict[str, Any],
    error_message: str,
    category: str = "invalid_input",
    client=None,
) -> Dict[str, Any]:
    """Suggest safe corrections for a failing n8n JSON payload."""
    try:
        safe_payload = redact_sensitive_data(payload)
        safe_error = redact_sensitive_data(error_message)
        safe_category = redact_sensitive_data(category)

        config = get_ai_config()

        if client is None and config.mock_mode:
            return _mock_payload_fix(safe_payload, safe_error)

        schema = PayloadFixOutput.model_json_schema()

        prompt = (
            "You are an n8n JSON payload repair assistant.\n"
            "Analyze the failing JSON payload and error message.\n"
            "Suggest a corrected payload when the fix is supported "
            "by the provided information.\n"
            "Never invent missing values.\n"
            "List unknown required values in needs_human_input.\n"
            "Return field-level changes with path, issue, and new_value.\n"
            "Never automatically execute or retry anything.\n"
            "Do not suggest retries for send_timeout, "
            "provider_rejection, or permanent_provider_rejection.\n"
            "Return JSON matching the supplied schema.\n\n"
            f"Error category: {safe_category}\n"
            f"Error message: {safe_error}\n"
            f"Payload: {json.dumps(safe_payload, ensure_ascii=False)}"
        )

        ai_client = client if client is not None else GeminiClient()
        response, _usage = ai_client.generate(
            prompt,
            schema=schema,
            temperature=0.1,
        )

        validated = PayloadFixOutput.model_validate(response)

        if safe_category in (
            "send_timeout",
            "provider_rejection",
            "permanent_provider_rejection",
        ):
            validated.retry_parameters = None

        return validated.model_dump()

    except Exception:
        logger.warning(
            "Payload fix suggestion failed; returning safe fallback."
        )
        return _fallback_payload_fix()
