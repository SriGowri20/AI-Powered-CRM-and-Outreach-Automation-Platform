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
