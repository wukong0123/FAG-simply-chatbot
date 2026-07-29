"""Embedding provider adapters for Gemini Cloud and local Ollama."""

from typing import Protocol

from config import Settings, settings
from ollama_client import OllamaClient


class EmbeddingServiceError(RuntimeError):
    """An actionable embedding provider error."""


class EmbeddingClient(Protocol):
    """Minimal interface used by ingest and retrieval."""

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """Create a batch of embedding vectors."""


class GeminiEmbeddingClient:
    """Generate embeddings with the managed Gemini API."""

    def __init__(self, config: Settings = settings) -> None:
        if not config.gemini_api_key:
            raise EmbeddingServiceError(
                "Thiếu GEMINI_API_KEY để tạo embedding cloud. "
                "Hãy cấu hình trong file .env."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise EmbeddingServiceError(
                "Chưa cài google-genai. Hãy chạy: pip install -r requirements.txt"
            ) from exc
        self.config = config
        self.types = types
        self.client = genai.Client(api_key=config.gemini_api_key)

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """Create Gemini embeddings with a stable configured dimension."""
        contents = [texts] if isinstance(texts, str) else texts
        try:
            response = self.client.models.embed_content(
                model=self.config.embedding_model,
                contents=contents,
                config=self.types.EmbedContentConfig(
                    output_dimensionality=self.config.embedding_dimensions
                ),
            )
            vectors = [embedding.values for embedding in response.embeddings or []]
            if len(vectors) != len(contents):
                raise EmbeddingServiceError(
                    "Số embedding Gemini trả về không khớp số văn bản đầu vào."
                )
            return vectors
        except EmbeddingServiceError:
            raise
        except Exception as exc:
            text = str(exc)
            if "429" in text:
                raise EmbeddingServiceError(
                    "Gemini Embedding đã vượt quota/rate limit. Vui lòng thử lại sau."
                ) from exc
            if "403" in text or "API key" in text:
                raise EmbeddingServiceError(
                    "GEMINI_API_KEY không hợp lệ hoặc chưa có quyền sử dụng model."
                ) from exc
            raise EmbeddingServiceError(f"Gemini Embedding trả về lỗi: {exc}") from exc


def create_embedding_client(config: Settings = settings) -> EmbeddingClient:
    """Build the configured embedding provider."""
    if config.embedding_provider == "gemini":
        return GeminiEmbeddingClient(config)
    return OllamaClient(config)
