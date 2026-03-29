"""Policy Retriever Agent: FAISS top-k with document_name and chunk_id."""

from utils.schemas import PolicyRetrieverOutput, RetrievedChunk
from rag import retriever as rag_retriever


def build_retrieval_query(ticket: str, classification: str) -> str:
    """Combine ticket and category for better semantic match."""
    return (
        f"Issue category: {classification}.\n"
        f"Customer message:\n{ticket}"
    )


def run_policy_retriever(
    ticket: str,
    classification: str,
    top_k: int | None = None,
) -> PolicyRetrieverOutput:
    """
    Retrieve top-k chunks from FAISS.
    Each chunk exposes document_name, chunk_id, doc (citation id), and text
    (e.g. refund_policy.txt_chunk_2).
    """
    q = build_retrieval_query(ticket, classification)
    return rag_retriever.retrieve(q, top_k=top_k)


def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """Human-readable block for downstream LLM (retrieved policy only)."""
    parts: list[str] = []
    for c in chunks:
        parts.append(
            f"[{c.chunk_id}] (document: {c.document_name})\n{c.text.strip()}",
        )
    return "\n\n---\n\n".join(parts) if parts else "(no policy chunks retrieved)"
