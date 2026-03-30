"""
Evaluation: 20 test cases — citation coverage, unsupported-claim proxy, escalation correctness.

Run from project root:
  python evaluation/run_evaluation.py
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.compliance_agent import run_compliance
from agents.policy_retriever_agent import run_policy_retriever
from agents.resolution_writer_agent import run_resolution_writer
from agents.triage_agent import run_triage
from utils.config import get_settings


def _load_cases() -> list[dict]:
    p = Path(__file__).parent / "test_cases.json"
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def run_one(case: dict) -> dict:
    ticket = case.get("ticket") or case.get("input") or ""
    ctx = case.get("order_context") or {}
    triage = run_triage(ticket, ctx)
    retrieval = run_policy_retriever(
        ticket=ticket,
        classification=triage.classification,
        top_k=None,
    )
    retrieved_ids = {c.chunk_id for c in retrieval.chunks}

    writer = run_resolution_writer(
        ticket=ticket,
        order_context=ctx,
        classification=triage.classification,
        triage_confidence=triage.confidence,
        clarifying_questions_from_triage=triage.clarifying_questions,
        chunks=retrieval.chunks,
    )

    compliance = run_compliance(
        ticket=ticket,
        order_context=ctx,
        chunks=retrieval.chunks,
        draft=writer,
    )

    citations = compliance.citations
    if citations:
        covered = sum(1 for c in citations if c in retrieved_ids)
        citation_coverage = covered / len(citations)
    else:
        # Empty citations allowed when escalating with no evidence
        citation_coverage = 1.0 if compliance.decision == "escalate" else 0.0

    unsupported = 0.0 if compliance.passed else 1.0

    exp_esc = case.get("expected_escalation")
    if exp_esc is None:
        exp_dict = case.get("expected")
        if isinstance(exp_dict, dict) and "decision" in exp_dict:
            exp_esc = exp_dict.get("decision") == "escalate"
    if exp_esc is None:
        escalation_correct = None
    else:
        got_esc = compliance.decision == "escalate"
        escalation_correct = 1.0 if (got_esc == bool(exp_esc)) else 0.0

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "classification": triage.classification,
        "decision": compliance.decision,
        "citation_coverage": citation_coverage,
        "unsupported_claim_proxy": unsupported,
        "escalation_correct": escalation_correct,
        "retrieved_chunk_count": len(retrieval.chunks),
        "citations": citations,
    }


def main():
    get_settings()  # loads .env
    cases = _load_cases()
    rows = []
    for c in cases:
        try:
            rows.append(run_one(c))
        except Exception as e:
            rows.append(
                {
                    "id": c.get("id"),
                    "error": str(e),
                },
            )

    cov = [r["citation_coverage"] for r in rows if "citation_coverage" in r]
    unsup = [r["unsupported_claim_proxy"] for r in rows if "unsupported_claim_proxy" in r]
    esc = [r["escalation_correct"] for r in rows if r.get("escalation_correct") is not None]

    report = {
        "n_cases": len(cases),
        "citation_coverage_rate": sum(cov) / len(cov) if cov else 0.0,
        "unsupported_claim_rate": sum(unsup) / len(unsup) if unsup else 0.0,
        "escalation_correctness": sum(esc) / len(esc) if esc else None,
        "per_case": rows,
    }

    out_path = Path(__file__).parent / "evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
