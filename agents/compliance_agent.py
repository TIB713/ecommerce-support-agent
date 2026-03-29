"""Compliance: citations, unsupported claims, policy violations — rewrite or escalate."""

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from utils.config import get_settings
from utils.json_utils import parse_llm_json_robust
from utils.schemas import ComplianceAgentOutput, ResolutionWriterOutput, RetrievedChunk


COMPLIANCE_SYSTEM = """You are a compliance checker for e-commerce support replies.

You are given:
- RETRIEVED POLICY CONTEXT (only allowed evidence)
- A draft resolution (decision, rationale, citations, customer_response)

Check:
1) Every citation in the draft must appear in the allowed chunk_id list and the cited content must support the claims.
2) No unsupported factual claims in rationale or customer_response relative to the context.
3) No policy violations (e.g. promising refunds not supported by text).

If fully compliant: action=accept, keep fields aligned with draft (may lightly polish wording without new facts).

If minor wording issues but facts OK: action=rewrite — fix customer_response/rationale to be strictly supported.

If missing citations, invented rules, or unsupported claims: action=escalate — set decision to escalate, explain briefly in internal_notes.

Output ONLY JSON with this shape:
{{
  "passed": boolean,
  "issues": string[],
  "action": "accept" | "rewrite" | "escalate",
  "decision": "approve"|"deny"|"partial"|"escalate"|"needs_info",
  "rationale": string,
  "citations": string[],
  "customer_response": string,
  "internal_notes": string
}}
"""


def run_compliance(
    ticket: str,
    order_context: dict[str, Any],
    chunks: list[RetrievedChunk],
    draft: ResolutionWriterOutput,
) -> ComplianceAgentOutput:
    s = get_settings()
    allowed_ids = [c.chunk_id for c in chunks]
    id_set = set(allowed_ids)

    deterministic_issues = _deterministic_check(chunks, draft)
    if deterministic_issues and draft.decision != "escalate":
        # If invalid citations, force escalate path unless already escalate
        invalid_cites = [c for c in draft.citations if c not in id_set]
        if invalid_cites:
            return ComplianceAgentOutput(
                passed=False,
                issues=deterministic_issues
                + [f"Invalid citations not in retrieval: {invalid_cites}"],
                decision="escalate",
                rationale=(
                    draft.rationale
                    + " [Compliance: citation mismatch — escalated for human review.]"
                ),
                citations=[c for c in draft.citations if c in id_set],
                customer_response=(
                    "We're escalating your request to ensure our response matches the official policy."
                ),
                internal_notes=draft.internal_notes
                + " | Compliance escalation: invalid or missing citations.",
                action="escalate",
            )

    if not chunks:
        return ComplianceAgentOutput(
            passed=False,
            issues=["No retrieved chunks"],
            decision="escalate",
            rationale="No policy evidence available; escalate.",
            citations=[],
            customer_response=(
                "We're connecting you with a specialist who can review your request."
            ),
            internal_notes="Compliance: empty retrieval.",
            action="escalate",
        )

    llm = ChatGroq(
        model=s.groq_model,
        api_key=s.groq_api_key,
        temperature=0.0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COMPLIANCE_SYSTEM),
            (
                "human",
                "Order context:\n{ctx}\n\nTicket:\n{ticket}\n\n"
                "RETRIEVED POLICY CONTEXT:\n{ctx_chunks}\n\n"
                "Allowed chunk_ids:\n{allowed}\n\n"
                "Draft JSON:\n{draft}\n",
            ),
        ],
    )
    chain = prompt | llm
    ctx = json.dumps(order_context, ensure_ascii=False, indent=2)
    ctx_chunks = _chunks_for_compliance(chunks)
    draft_json = json.dumps(draft.model_dump(), ensure_ascii=False, indent=2)
    msg = chain.invoke(
        {
            "ctx": ctx,
            "ticket": ticket,
            "ctx_chunks": ctx_chunks,
            "allowed": json.dumps(allowed_ids),
            "draft": draft_json,
        },
    )
    text = msg.content if hasattr(msg, "content") else str(msg)
    try:
        data = parse_llm_json_robust(text)
    except ValueError:
        return ComplianceAgentOutput(
            passed=False,
            issues=["Compliance JSON parse failure"],
            decision="escalate",
            rationale=draft.rationale + " [Compliance: invalid model JSON — escalated.]",
            citations=[c for c in draft.citations if c in id_set],
            customer_response=(
                "We're escalating your request to ensure our response matches the official policy."
            ),
            internal_notes=draft.internal_notes + " | Compliance parse failure.",
            action="escalate",
        )

    action = str(data.get("action", "accept")).lower().strip()
    if action not in {"accept", "rewrite", "escalate"}:
        action = "accept"

    decision = str(data.get("decision", draft.decision)).lower().strip()
    if decision not in {"approve", "deny", "partial", "escalate", "needs_info"}:
        decision = draft.decision

    citations = data.get("citations")
    if not isinstance(citations, list):
        citations = draft.citations
    citations = [str(x).strip() for x in citations if str(x).strip()]
    citations = [c for c in citations if c in id_set]

    out = ComplianceAgentOutput(
        passed=bool(data.get("passed", False)),
        issues=[str(x) for x in (data.get("issues") or [])] if isinstance(data.get("issues"), list) else [],
        decision=decision,  # type: ignore[arg-type]
        rationale=str(data.get("rationale", draft.rationale)),
        citations=citations,
        customer_response=str(data.get("customer_response", draft.customer_response)),
        internal_notes=str(data.get("internal_notes", draft.internal_notes)),
        action=action,  # type: ignore[arg-type]
    )

    # Final hard gate
    if out.decision in {"approve", "deny", "partial"} and not out.citations:
        out = ComplianceAgentOutput(
            passed=False,
            issues=out.issues + ["Missing citations for non-escalate decision"],
            decision="escalate",
            rationale=out.rationale + " [Compliance: missing citations.]",
            citations=[],
            customer_response=(
                "We're escalating your request so a specialist can confirm the correct policy outcome."
            ),
            internal_notes=out.internal_notes + " | Forced escalate: citations empty.",
            action="escalate",
        )

    return out


def _chunks_for_compliance(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"[{c.chunk_id}] {c.document_name}\n{c.text}")
    return "\n\n".join(lines)


def _deterministic_check(
    chunks: list[RetrievedChunk],
    draft: ResolutionWriterOutput,
) -> list[str]:
    issues: list[str] = []
    allowed = {c.chunk_id for c in chunks}
    for cid in draft.citations:
        if cid not in allowed:
            issues.append(f"Unknown citation: {cid}")
    return issues
