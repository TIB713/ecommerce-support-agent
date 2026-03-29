"""Document cleaning, chunking, and metadata (chunk ids)."""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(raw: str) -> str:
    """Normalize whitespace; strip noise common in policy dumps."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def document_name_from_path(path: Path) -> str:
    return path.name


def stable_file_stem(path: Path) -> str:
    """Legacy stem slug for metadata (search/analytics)."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", path.stem).strip("_").lower() or "document"


def safe_document_filename(name: str) -> str:
    """Filename safe for citation ids: e.g. refund_policy.txt."""
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "").strip()).strip("_")
    return s or "document.txt"


def load_policy_documents(policies_dir: Path) -> list[Document]:
    """Load all .txt and .md policy files."""
    docs: list[Document] = []
    if not policies_dir.is_dir():
        return docs
    for path in sorted(policies_dir.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_text(raw)
        stem = stable_file_stem(path)
        docs.append(
            Document(
                page_content=cleaned,
                metadata={
                    "source_path": str(path),
                    "document_name": document_name_from_path(path),
                    "file_stem": stem,
                },
            )
        )
    return docs


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split with overlap; assign chunk_id per chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    out: list[Document] = []
    for doc in documents:
        meta = dict(doc.metadata)
        fname = safe_document_filename(str(meta.get("document_name", "document.txt")))
        pieces = splitter.split_text(doc.page_content)
        for i, piece in enumerate(pieces):
            # Citation id: refund_policy.txt_chunk_2 (includes source filename)
            cid = f"{fname}_chunk_{i}"
            meta = dict(doc.metadata)
            meta["chunk_id"] = cid
            meta["chunk_index"] = i
            out.append(Document(page_content=piece, metadata=meta))
    return out
