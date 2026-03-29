"""Application configuration from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_project_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    huggingfacehub_api_token: str | None = None

    groq_model: str = "llama-3.3-70b-versatile"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    policies_dir: Path = _project_root() / "data" / "policies"
    faiss_index_dir: Path = _project_root() / "data" / "faiss_index"

    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5

    min_evidence_chunks: int = 1


def get_settings() -> Settings:
    return Settings()
