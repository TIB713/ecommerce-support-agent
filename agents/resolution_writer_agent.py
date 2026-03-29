"""Resolution Writer: Groq LLM, strict use of retrieved policy text only."""

import json
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from utils.config import get_settings
from utils.json_utils import parse_llm_json_robust
from utils.schemas import ResolutionWriterOutput, RetrievedChunk

WRITER_SYSTEM = """You are an e-commerce policy resolution writer. Output must be JSON ONLY — no markdown, no prose before or after the JSON object.

STRICT RULES:
1) Use ONLY the RETRIEVED POLICY CONTEXT below. Do NOT invent policies, time windows, fees, or exceptions not written there.
2) In "rationale", name the governing rule in plain language (e.g. which section or condition applies) and WHY that leads to the decision. Quote concepts, not invented numbers.
3) "citations" MUST list only chunk_id strings copied exactly from the ALLOWED list. Format looks like: filename.txt_chunk_0
4) If CONTEXT_SUFFICIENT is true and the policy text supports a return/refund for this scenario, prefer decision "approve" (or "partial" if policy says partial) with citations — do NOT use "needs_info" unless the policy itself requires verification that cannot be inferred from the ticket and order JSON.
5) Use "needs_info" ONLY when the policy requires specific missing facts (e.g. item not described) AND those facts are not in the ticket or order JSON.
6) If retrieved policy does not cover the issue or is irrelevant, decision MUST be "escalate" with empty citations.
7) If minimum evidence is not met (caller will escalate separately), still respond with valid JSON; prefer "escalate" if context is empty.

Required JSON keys exactly:
decision, rationale, citations, customer_response, internal_notes

decision must be one of: approve | deny | partial | escalate | needs_info"""


def run_resolution_writer(
    ticket: str,
    order_context: dict[str, Any],
    classification: str,
    triage_confidence: float,
    clarifying_questions_from_triage: list[str],
    chunks: list[RetrievedChunk],
    *,
    context_sufficient: bool = False,
) -> ResolutionWriterOutput:
    s = get_settings()
    allowed_ids = {c.chunk_id for c in chunks}
    context_block = _format_context(chunks)

    if len(chunks) < s.min_evidence_chunks:
        return ResolutionWriterOutput(
            decision="escalate",
            rationale=(
                "No sufficient policy evidence retrieved (below minimum evidence threshold); "
                "escalate per policy-grounding rules."
            ),
            citations=[],
            customer_response=(
                "We're escalating your request to a specialist who can review it with the "
                "correct policy details."
            ),
            internal_notes="Escalated: insufficient retrieved policy chunks.",
        )

    llm = ChatGroq(
        model=s.groq_model,
        api_key=s.groq_api_key,
        temperature=0.1,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", WRITER_SYSTEM),
            (
                "human",
                "CONTEXT_SUFFICIENT: {context_sufficient}\n"
                "Classification: {classification}\n"
                "Triage confidence: {triage_confidence}\n"
                "Optional clarifying questions from triage (only if still needed):\n{triage_qs}\n\n"
                "Order context (JSON):\n{ctx}\n\n"
                "Ticket:\n{ticket}\n\n"
                "RETRIEVED POLICY CONTEXT (only source of truth):\n{context_block}\n\n"
                "Allowed chunk_id values (citations MUST be from this list only):\n{allowed}\n",
            ),
        ],
    )
    chain = prompt | llm
    ctx = json.dumps(order_context, ensure_ascii=False, indent=2)
    triage_qs = json.dumps(clarifying_questions_from_triage, ensure_ascii=False)
    payload = {
        "context_sufficient": "true" if context_sufficient else "false",
        "classification": classification,
        "triage_confidence": triage_confidence,
        "triage_qs": triage_qs,
        "ctx": ctx,
        "ticket": ticket,
        "context_block": context_block,
        "allowed": json.dumps(sorted(allowed_ids)),
    }
    msg = chain.invoke(payload)
    text = msg.content if hasattr(msg, "content") else str(msg)

    data: dict[str, Any] | None = None
    try:
        data = parse_llm_json_robust(text)
    except (ValueError, json.JSONDecodeError):
        repair_llm = ChatGroq(
            model=s.groq_model,
            api_key=s.groq_api_key,
            temperature=0.0,
        )
        fix = repair_llm.invoke(
            [
                HumanMessage(
                    content=(
                        "Return ONLY one valid JSON object with keys: "
                        "decision, rationale, citations, customer_response, internal_notes. "
                        "No markdown. Fix/complete from this model output:\n\n"
                        + text[:6000]
                    ),
                ),
            ],
        )
        fix_text = fix.content if hasattr(fix, "content") else str(fix)
        try:
            data = parse_llm_json_robust(fix_text)
        except (ValueError, json.JSONDecodeError):
            return ResolutionWriterOutput(
                decision="escalate",
                rationale="Model output was not valid JSON after repair attempt; escalate.",
                citations=[],
                customer_response=(
                    "We're escalating your request so a specialist can confirm the correct policy."
                ),
                internal_notes="Writer JSON parse failure; escalated.",
            )

    assert data is not None
    decision = str(data.get("decision", "escalate")).lower().strip()
    if decision not in {"approve", "deny", "partial", "escalate", "needs_info"}:
        decision = "escalate"

    rationale = str(data.get("rationale", "")).strip()
    citations = data.get("citations") or []
    if not isinstance(citations, list):
        citations = []
    citations = [str(x).strip() for x in citations if str(x).strip()]

    customer_response = str(data.get("customer_response", "")).strip()
    internal_notes = str(data.get("internal_notes", "")).strip()

    citations = [c for c in citations if c in allowed_ids]

    if decision in {"approve", "deny", "partial"} and not citations:
        decision = "escalate"
        rationale = (
            (rationale + " ") if rationale else ""
        ) + "[Auto-escalated: no valid citations from retrieved policy.]"
        customer_response = (
            "We need to escalate your case so our team can confirm the applicable policy."
        )

    if (
        context_sufficient
        and classification == "refund"
        and decision == "needs_info"
        and citations
    ):
        decision = "approve"
        rationale = (
            "According to the cited policy, the described situation matches eligible return/refund "
            "conditions; proceeding with approval. "
            + rationale
        )

    return ResolutionWriterOutput(
        decision=decision,  # type: ignore[arg-type]
        rationale=rationale,
        citations=citations,
        customer_response=customer_response,
        internal_notes=internal_notes,
    )


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"chunk_id: {c.chunk_id}\nsource_file: {c.document_name}\ndoc: {c.doc}\n{c.text}",
        )
    return "\n\n---\n\n".join(parts)
