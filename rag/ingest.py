"""Build FAISS index from policy documents."""

from pathlib import Path

from langchain_community.vectorstores import FAISS

from rag.embeddings import build_embeddings
from rag.pipeline import chunk_documents, load_policy_documents
from utils.config import get_settings


def ingest_policies(
    policies_dir: Path | None = None,
    index_dir: Path | None = None,
) -> dict:
    """
    Full pipeline: load -> clean (in loader) -> chunk -> embed -> FAISS save.
    """
    s = get_settings()
    policies_dir = policies_dir or s.policies_dir
    index_dir = index_dir or s.faiss_index_dir

    raw_docs = load_policy_documents(policies_dir)
    if not raw_docs:
        raise FileNotFoundError(f"No .txt/.md policies under {policies_dir}")

    chunks = chunk_documents(
        raw_docs,
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
    )

    embeddings = build_embeddings()
    index_dir.mkdir(parents=True, exist_ok=True)

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(index_dir))

    return {
        "policies_dir": str(policies_dir),
        "index_dir": str(index_dir),
        "source_documents": len(raw_docs),
        "chunks": len(chunks),
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
    }
