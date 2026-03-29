"""HuggingFace embeddings (free sentence-transformers model)."""

from utils.config import get_settings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings


def build_embeddings():
    s = get_settings()
    kwargs: dict = {
        "model_name": s.hf_embedding_model,
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True},
    }
    if s.huggingfacehub_api_token:
        kwargs["model_kwargs"]["token"] = s.huggingfacehub_api_token
    return HuggingFaceEmbeddings(**kwargs)
