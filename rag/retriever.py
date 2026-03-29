"""Load FAISS and retrieve top-k with document_name + chunk_id."""

from pathlib import Path

from langchain_community.vectorstores import FAISS

from rag.embeddings import build_embeddings
from utils.config import get_settings
from utils.schemas import PolicyRetrieverOutput, RetrievedChunk


def load_vectorstore(index_dir: Path | None = None) -> FAISS:
    s = get_settings()
    index_dir = index_dir or s.faiss_index_dir
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_dir}. Run POST /ingest first.",
        )
    embeddings = build_embeddings()
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve(
    query: str,
    top_k: int | None = None,
    index_dir: Path | None = None,
) -> PolicyRetrieverOutput:
    s = get_settings()
    k = top_k if top_k is not None else s.top_k
    vs = load_vectorstore(index_dir)
    pairs = vs.similarity_search_with_score(query, k=k)
    chunks: list[RetrievedChunk] = []
    for doc, score in pairs:
        meta = doc.metadata or {}
        doc_name = meta.get("document_name", "unknown")
        cid = meta.get("chunk_id", "unknown_chunk")
        cid_s = str(cid)
        chunks.append(
            RetrievedChunk(
                document_name=str(doc_name),
                chunk_id=cid_s,
                doc=cid_s,
                text=doc.page_content,
                score=float(score) if score is not None else None,
            ),
        )
    return PolicyRetrieverOutput(chunks=chunks, query_used=query)
