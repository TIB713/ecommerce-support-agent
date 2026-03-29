"""Heuristics: when ticket + order_context are enough to decide without extra questions."""

import re
from typing import Any


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def refine_clarifying_questions(
    ticket: str,
    order_context: dict[str, Any],
    questions: list[str],
    classification: str,
) -> list[str]:
    """
    Drop questions that duplicate information already stated in the ticket
    or present in structured order_context.
    """
    t = _norm(ticket)
    out: list[str] = []
    od = str(order_context.get("order_date") or "")
    dd = str(order_context.get("delivery_date") or "")
    ost = str(order_context.get("order_status") or "")

    for q in (questions or [])[:5]:
        ql = _norm(q)
        if not ql:
            continue
        # Item condition already described
        if any(
            k in ql
            for k in (
                "packaging",
                "unopened",
                "opened",
                "condition",
                "sealed",
            )
        ):
            if any(
                x in t
                for x in (
                    "unopened",
                    "unused",
                    "sealed",
                    "never opened",
                    "original packaging",
                    "brand new",
                    "still in the box",
                )
            ):
                continue
        # Receipt / order number when we already have order context
        if any(k in ql for k in ("receipt", "order number", "proof of purchase")):
            if od or ost or re.search(r"\b(order|#\d)\b", t):
                continue
        # Delivery timing
        if any(k in ql for k in ("delivered", "delivery", "arrived", "received")):
            if dd or any(x in t for x in ("delivered", "arrived", "received", "yesterday", "today")):
                continue
        # Structured status
        if "order_status" in ql or "status" in ql:
            if ost:
                continue

        out.append(q.strip())

    return out[:3]


def is_context_sufficient_for_decision(
    ticket: str,
    order_context: dict[str, Any],
    classification: str,
) -> bool:
    """
    True when we already have enough facts for a policy-grounded refund/return style
    decision without asking the customer for more (per assignment examples).
    """
    t = _norm(ticket)
    ost = str(order_context.get("order_status") or "").lower()
    dd = str(order_context.get("delivery_date") or "").strip()

    if classification == "refund":
        item_ok = any(
            x in t
            for x in (
                "unopened",
                "unused",
                "sealed",
                "never opened",
                "original packaging",
                "not used",
            )
        )
        delivery_ok = bool(dd) or any(
            x in t for x in ("delivered", "arrived", "received", "yesterday", "today", "last week")
        )
        order_ok = bool(ost) or bool(order_context.get("order_date")) or "order" in t
        return bool(item_ok and delivery_ok and order_ok)

    # Other categories: do not force "sufficient" — triage + writer handle questions.
    return False
