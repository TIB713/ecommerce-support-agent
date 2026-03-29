"""Triage: classify issue, confidence, missing fields, max 3 clarifying questions."""

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from utils.config import get_settings
from utils.json_utils import parse_llm_json_robust
from utils.schemas import TriageAgentOutput


TRIAGE_SYSTEM = """You are a senior e-commerce support triage model.

Classify the customer issue into exactly ONE label:
refund | shipping | payment | promo | fraud | other

Label rules:
- refund: returns, refunds, exchanges, refund eligibility
- shipping: delivery delays, tracking, carriers, address changes
- payment: card declines, wallet, installments, billing
- promo: coupons, discounts, loyalty credits, price matches
- fraud: suspected abuse, account takeover, chargeback abuse
- other: anything else

CLARIFYING QUESTIONS (critical):
- Ask at most 3 short questions ONLY if information REQUIRED to route or decide is MISSING from the ticket AND from the order context JSON.
- If the ticket already states item condition (e.g. unopened), delivery timing, and the order context includes order_status / delivery_date where relevant, use an EMPTY clarifying_questions array.
- Do NOT ask about facts the customer already gave in the ticket (e.g. do not ask "is it unopened?" if they said unopened).
- Prefer questions about: order_status, delivery_date, item_condition — only if still unknown.

missing_fields: short labels for what is missing (e.g. "delivery_date") — can be empty if nothing material is missing.

Output ONLY valid JSON with keys:
- classification (string, one of the six)
- confidence (number 0-1)
- missing_fields (array of strings)
- clarifying_questions (array, at most 3 strings; empty array if not needed)

No markdown, no text outside JSON."""


def run_triage(
    ticket: str,
    order_context: dict[str, Any],
) -> TriageAgentOutput:
    s = get_settings()
    llm = ChatGroq(
        model=s.groq_model,
        api_key=s.groq_api_key,
        temperature=0.0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", TRIAGE_SYSTEM),
            (
                "human",
                "Ticket:\n{ticket}\n\nOrder context (JSON):\n{ctx}\n",
            ),
        ],
    )
    chain = prompt | llm
    ctx = json.dumps(order_context, ensure_ascii=False, indent=2)
    msg = chain.invoke({"ticket": ticket, "ctx": ctx})
    text = msg.content if hasattr(msg, "content") else str(msg)

    try:
        data = parse_llm_json_robust(text)
    except ValueError:
        data = {
            "classification": "other",
            "confidence": 0.5,
            "missing_fields": [],
            "clarifying_questions": [],
        }

    cls = str(data.get("classification", "other")).lower().strip()
    allowed = {"refund", "shipping", "payment", "promo", "fraud", "other"}
    if cls not in allowed:
        cls = "other"

    conf = data.get("confidence", 0.5)
    try:
        confidence = float(conf)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    missing = data.get("missing_fields") or []
    if not isinstance(missing, list):
        missing = []
    missing = [str(x) for x in missing][:10]

    qs = data.get("clarifying_questions") or []
    if not isinstance(qs, list):
        qs = []
    qs = [str(q).strip() for q in qs if str(q).strip()][:3]

    return TriageAgentOutput(
        classification=cls,  # type: ignore[arg-type]
        confidence=confidence,
        missing_fields=missing,
        clarifying_questions=qs,
    )
