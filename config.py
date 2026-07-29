"""Application configuration loaded from environment variables."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} phải là số nguyên, nhận được: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} phải lớn hơn 0.")
    return parsed


def _threshold(name: str, default: float) -> float:
    value = os.getenv(name, str(default))
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} phải là số, nhận được: {value!r}") from exc
    if not 0 <= parsed <= 1:
        raise ValueError(f"{name} phải nằm trong khoảng 0 đến 1.")
    return parsed


def _project_path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings."""

    ollama_base_url: str
    ollama_api_key: str
    chat_model: str
    embedding_provider: str
    embedding_model: str
    gemini_api_key: str
    embedding_dimensions: int
    top_k: int
    similarity_threshold: float
    faq_data_path: Path
    embeddings_path: Path
    metadata_path: Path
    request_timeout_seconds: int


def load_settings() -> Settings:
    """Load settings with safe defaults and validation."""
    base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com").rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("OLLAMA_BASE_URL phải bắt đầu bằng http:// hoặc https://.")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "gemini").strip().lower()
    if embedding_provider not in {"gemini", "ollama"}:
        raise ValueError("EMBEDDING_PROVIDER phải là 'gemini' hoặc 'ollama'.")
    return Settings(
        ollama_base_url=base_url,
        ollama_api_key=os.getenv("OLLAMA_API_KEY", "").strip(),
        chat_model=os.getenv("OLLAMA_CHAT_MODEL", "gpt-oss:20b").strip(),
        embedding_provider=embedding_provider,
        embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        ).strip()
        if embedding_provider == "gemini"
        else os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        embedding_dimensions=_positive_int("EMBEDDING_DIMENSIONS", 768),
        top_k=_positive_int("TOP_K", 3),
        similarity_threshold=_threshold("SIMILARITY_THRESHOLD", 0.55),
        faq_data_path=_project_path("FAQ_DATA_PATH", "data/faqs.json"),
        embeddings_path=_project_path("EMBEDDINGS_PATH", "storage/embeddings.npy"),
        metadata_path=_project_path("METADATA_PATH", "storage/metadata.json"),
        request_timeout_seconds=_positive_int("REQUEST_TIMEOUT_SECONDS", 120),
    )


settings = load_settings()
