"""Validate API resolution payload; safe fallback when validation fails."""

from typing import Any, Literal

from pydantic import ValidationError

from utils.schemas import ResolutionOutput

Decision = Literal["approve", "deny", "partial", "escalate", "needs_info"]
VALID_DECISIONS: frozenset[str] = frozenset(
    {"approve", "deny", "partial", "escalate", "needs_info"},
)


def validate_resolution_payload(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (ok, error messages)."""
    errors: list[str] = []
    try:
        ResolutionOutput.model_validate(data)
    except ValidationError as e:
        return False, [str(e)]

    d = data.get("decision")
    if d not in VALID_DECISIONS:
        errors.append(f"invalid decision: {d!r}")

    # Citations required for binding approvals/denials/partials
    cites = data.get("citations") or []
    if d in {"approve", "deny", "partial"}:
        if not isinstance(cites, list) or len(cites) == 0:
            errors.append("citations empty for non-escalate decision")

    conf = data.get("confidence")
    if not isinstance(conf, (int, float)):
        errors.append("confidence must be a number")

    return len(errors) == 0, errors


def fallback_escalation_payload(
    *,
    classification: str,
    confidence: float,
    clarifying_questions: list[str],
    reason: str,
) -> dict[str, Any]:
    """Safe JSON-serializable dict when pipeline or validation fails."""
    base = {
        "classification": classification,
        "confidence": round(float(confidence), 2),
        "clarifying_questions": clarifying_questions[:3],
        "decision": "escalate",
        "rationale": (
            f"System could not produce a validated policy-grounded response ({reason}). "
            "Escalated for human review."
        ),
        "citations": [],
        "customer_response": (
            "We’re escalating your request to a specialist who can review it and respond "
            "with the correct policy details."
        ),
        "internal_notes": f"Auto-escalation fallback: {reason}",
    }
    return base
