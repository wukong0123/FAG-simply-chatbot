"""Shared data validation and storage helpers."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class DataValidationError(ValueError):
    """Raised when FAQ input is invalid."""


def validate_faqs(raw: Any) -> list[dict[str, Any]]:
    """Validate and normalize a decoded FAQ list."""
    if not isinstance(raw, list):
        raise DataValidationError("Dữ liệu FAQ phải là một danh sách JSON.")
    result: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise DataValidationError(f"FAQ ở vị trí {index} phải là một object.")
        faq_id = item.get("id")
        if faq_id is None:
            raise DataValidationError(f"FAQ ở vị trí {index} thiếu id.")
        if faq_id in seen_ids:
            raise DataValidationError(f"FAQ id {faq_id!r} bị trùng.")
        seen_ids.add(faq_id)
        question = item.get("question")
        answer = item.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise DataValidationError(f"FAQ id {faq_id!r}: question không được rỗng.")
        if not isinstance(answer, str) or not answer.strip():
            raise DataValidationError(f"FAQ id {faq_id!r}: answer không được rỗng.")
        category = item.get("category", "general")
        if not isinstance(category, str) or not category.strip():
            category = "general"
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias.strip() for alias in aliases
        ):
            raise DataValidationError(
                f"FAQ id {faq_id!r}: aliases phải là danh sách chuỗi không rỗng."
            )
        normalized_aliases = list(
            dict.fromkeys(" ".join(alias.split()) for alias in aliases)
        )
        normalized = {
            "id": faq_id,
            "question": " ".join(question.split()),
            "answer": " ".join(answer.split()),
            "category": " ".join(category.split()),
        }
        if normalized_aliases:
            normalized["aliases"] = normalized_aliases
        result.append(normalized)
    if not result:
        raise DataValidationError("Danh sách FAQ không được rỗng.")
    return result


def load_faqs(path: Path) -> list[dict[str, Any]]:
    """Read and validate UTF-8 FAQ data."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataValidationError(f"Không tìm thấy file FAQ: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataValidationError(
            f"JSON không hợp lệ tại dòng {exc.lineno}, cột {exc.colno}: {exc.msg}"
        ) from exc
    return validate_faqs(raw)


def faq_content_hash(faqs: list[dict[str, Any]]) -> str:
    """Return a stable SHA-256 hash of normalized FAQ content."""
    encoded = json.dumps(
        faqs, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector, rejecting empty and zero vectors."""
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Embedding phải là vector một chiều và không rỗng.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Embedding chứa giá trị không hợp lệ.")
    norm = float(np.linalg.norm(array))
    if norm == 0:
        raise ValueError("Embedding không được là vector 0.")
    return array / norm


def cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Compute vectorized cosine scores for normalized or raw vectors."""
    vectors = np.asarray(matrix, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError("Ma trận embedding phải có shape (n, d) và không rỗng.")
    query_vector = normalize_vector(query)
    if vectors.shape[1] != query_vector.shape[0]:
        raise ValueError("Số chiều query embedding không khớp dữ liệu lưu trữ.")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("Dữ liệu lưu trữ chứa vector 0.")
    return (vectors @ query_vector) / norms
