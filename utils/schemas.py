"""Pydantic models for API contracts and agent I/O."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


OrderContext = dict[str, Any]


class OrderContextModel(BaseModel):
    """Structured order context (optional fields for flexibility)."""

    order_date: str | None = None
    delivery_date: str | None = None
    item_category: str | None = None
    fulfillment_type: str | None = None
    shipping_region: str | None = None
    order_status: str | None = None
    payment_method: str | None = None

    model_config = {"extra": "allow"}


class ResolutionOutput(BaseModel):
    """Strict JSON output for /query and Streamlit."""

    classification: str = Field(
        ...,
        description="Issue category from triage",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Triage confidence 0.0–1.0",
    )
    clarifying_questions: list[str] = Field(default_factory=list)
    decision: Literal["approve", "deny", "partial", "escalate", "needs_info"] = Field(
        ...,
    )
    rationale: str = Field(..., description="Explanation grounded in policy citations")
    citations: list[str] = Field(
        ...,
        description="Citation ids: filename.ext_chunk_n e.g. refund_policy.txt_chunk_2",
    )
    customer_response: str = Field(..., description="Customer-facing reply")
    internal_notes: str = Field(default="", description="Internal agent notes")


class TriageAgentOutput(BaseModel):
    classification: Literal[
        "refund",
        "shipping",
        "payment",
        "promo",
        "fraud",
        "other",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list, max_length=3)


class RetrievedChunk(BaseModel):
    """Structured retriever row: citation id + source filename + text."""

    document_name: str
    chunk_id: str
    doc: str = Field(
        default="",
        description="Citation id (defaults to chunk_id): filename.ext_chunk_index",
    )
    text: str
    score: float | None = None

    @model_validator(mode="after")
    def _doc_matches_chunk(self):
        if not self.doc:
            object.__setattr__(self, "doc", self.chunk_id)
        return self


class PolicyRetrieverOutput(BaseModel):
    chunks: list[RetrievedChunk]
    query_used: str


class ResolutionWriterOutput(BaseModel):
    decision: Literal["approve", "deny", "partial", "escalate", "needs_info"]
    rationale: str
    citations: list[str]
    customer_response: str
    internal_notes: str


class ComplianceAgentOutput(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    decision: Literal["approve", "deny", "partial", "escalate", "needs_info"]
    rationale: str
    citations: list[str]
    customer_response: str
    internal_notes: str
    action: Literal["accept", "rewrite", "escalate"]
