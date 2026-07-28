"""Semantic retrieval over locally stored NumPy embeddings."""

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import Settings, settings
from ollama_client import OllamaClient
from utils import cosine_scores, faq_content_hash, load_faqs


class StorageError(RuntimeError):
    """Embedding storage is absent, stale, or invalid."""


class Retriever:
    """Retrieve semantically similar FAQ records."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        config: Settings = settings,
        embed_fn: Callable[[str], list[float] | np.ndarray] | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.embed_fn = embed_fn
        self.embeddings, self.faqs = self._load_storage()

    def _load_storage(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if not self.config.embeddings_path.exists() or not self.config.metadata_path.exists():
            raise StorageError(
                "Chưa tìm thấy dữ liệu embedding.\nHãy chạy: python ingest.py"
            )
        try:
            matrix = np.load(self.config.embeddings_path, allow_pickle=False)
            metadata = json.loads(
                self.config.metadata_path.read_text(encoding="utf-8")
            )
            faqs = metadata["faqs"]
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise StorageError(f"Dữ liệu embedding bị hỏng: {exc}") from exc
        if matrix.ndim != 2 or matrix.shape[0] != len(faqs):
            raise StorageError("Số lượng metadata không khớp số vector.")
        current_faqs = load_faqs(self.config.faq_data_path)
        if metadata.get("content_hash") != faq_content_hash(current_faqs):
            raise StorageError(
                "Dữ liệu FAQ đã thay đổi so với lần ingest gần nhất.\n"
                "Hãy chạy lại: python ingest.py"
            )
        return matrix, faqs

    def _embed_query(self, question: str) -> np.ndarray:
        if self.embed_fn:
            return np.asarray(self.embed_fn(question), dtype=np.float32)
        service = self.client or OllamaClient(self.config)
        return np.asarray(service.embed(question)[0], dtype=np.float32)

    def retrieve(
        self, question: str, top_k: int | None = None
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return ranked matches and whether the query should be rejected."""
        clean_question = " ".join(question.split())
        if not clean_question:
            raise ValueError("Vui lòng nhập câu hỏi.")
        scores = cosine_scores(self.embeddings, self._embed_query(clean_question))
        limit = min(top_k or self.config.top_k, len({faq["id"] for faq in self.faqs}))
        matches = []
        seen_ids: set[Any] = set()
        for index in np.argsort(scores)[::-1]:
            faq = self.faqs[int(index)]
            if faq["id"] in seen_ids:
                continue
            seen_ids.add(faq["id"])
            matches.append({**faq, "score": float(scores[int(index)])})
            if len(matches) == limit:
                break
        rejected = not matches or matches[0]["score"] < self.config.similarity_threshold
        return matches, rejected
