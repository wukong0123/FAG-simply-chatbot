"""Build and persist FAQ embeddings."""

import json
import logging
import time

import numpy as np

from config import settings
from ollama_client import OllamaClient, OllamaServiceError
from utils import faq_content_hash, load_faqs, normalize_vector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def ingest(client: OllamaClient | None = None) -> tuple[int, int]:
    """Validate FAQs, create embeddings, and save storage files."""
    started = time.perf_counter()
    faqs = load_faqs(settings.faq_data_path)
    LOGGER.info("Đã đọc %d FAQ.", len(faqs))
    texts = []
    embedding_records = []
    for faq in faqs:
        variants = [faq["question"], *faq.get("aliases", [])]
        for variant in variants:
            texts.append(f"Câu hỏi: {variant}\nChủ đề: {faq['category']}")
            embedding_records.append({**faq, "matched_text": variant})
    service = client or OllamaClient()
    vectors = service.embed(texts)
    normalized = [normalize_vector(np.asarray(vector)) for vector in vectors]
    dimensions = {vector.shape[0] for vector in normalized}
    if len(normalized) != len(embedding_records):
        raise ValueError("Số embedding trả về không khớp số văn bản FAQ và alias.")
    if len(dimensions) != 1:
        raise ValueError("Các embedding không có cùng số chiều.")
    matrix = np.vstack(normalized).astype(np.float32)
    settings.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(settings.embeddings_path, matrix, allow_pickle=False)
    metadata = {
        "version": 1,
        "content_hash": faq_content_hash(faqs),
        "embedding_model": settings.embedding_model,
        "dimensions": matrix.shape[1],
        "faqs": embedding_records,
    }
    settings.metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Model embedding: %s", settings.embedding_model)
    LOGGER.info("Số chiều vector: %d", matrix.shape[1])
    LOGGER.info("Số vector FAQ và alias: %d", matrix.shape[0])
    LOGGER.info("Đã lưu: %s", settings.embeddings_path)
    LOGGER.info("Đã lưu: %s", settings.metadata_path)
    LOGGER.info("Hoàn thành trong %.2f giây.", time.perf_counter() - started)
    return matrix.shape


if __name__ == "__main__":
    try:
        ingest()
        print("Ingest thành công.")
    except (ValueError, OSError, OllamaServiceError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from None
