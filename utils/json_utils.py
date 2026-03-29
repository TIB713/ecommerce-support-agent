"""Safe JSON extraction from LLM responses with validation and light repair."""

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating markdown fences."""
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if fence:
        s = fence.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    raw = s[start : end + 1]
    return json.loads(raw)


def parse_llm_json_robust(text: str) -> dict[str, Any]:
    """
    Try multiple strategies; raise last error if all fail.
    Use before retrying the LLM call.
    """
    try:
        return extract_json_object(text)
    except (json.JSONDecodeError, ValueError):
        pass
    s = (text or "").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Strip trailing commas (common LLM mistake)
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", s)
        start = fixed.find("{")
        end = fixed.rfind("}")
        if start != -1 and end > start:
            return json.loads(fixed[start : end + 1])
    except json.JSONDecodeError:
        pass
    raise ValueError("Could not parse JSON from model output")


def parse_llm_json_with_retry(
    text: str,
    repair_fn: Any | None = None,
) -> dict[str, Any]:
    """
    Parse JSON; optional repair_fn(text) -> new_text for one retry (e.g. second LLM call).
    """
    try:
        return parse_llm_json_robust(text)
    except (json.JSONDecodeError, ValueError) as e:
        if repair_fn is None:
            raise
        repaired = repair_fn(str(e))
        return parse_llm_json_robust(repaired)
