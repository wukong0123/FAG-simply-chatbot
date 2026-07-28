import json
from dataclasses import replace

import numpy as np
import pytest

from config import settings
from retriever import Retriever, StorageError
from utils import faq_content_hash


def make_config(tmp_path, threshold=0.5):
    faq_path = tmp_path / "faqs.json"
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.json"
    faqs = [
        {"id": 1, "question": "A", "answer": "AA", "category": "general"},
        {"id": 2, "question": "B", "answer": "BB", "category": "general"},
    ]
    faq_path.write_text(json.dumps(faqs), encoding="utf-8")
    np.save(embeddings_path, np.array([[1, 0], [0, 1]], dtype=np.float32))
    metadata_path.write_text(
        json.dumps({"faqs": faqs, "content_hash": faq_content_hash(faqs)}),
        encoding="utf-8",
    )
    return replace(
        settings,
        faq_data_path=faq_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        similarity_threshold=threshold,
    )


def test_top_k_order_without_real_ollama(tmp_path):
    retriever = Retriever(config=make_config(tmp_path), embed_fn=lambda _: [0.9, 0.1])
    matches, rejected = retriever.retrieve("query", top_k=2)
    assert [item["id"] for item in matches] == [1, 2]
    assert not rejected


def test_threshold_rejects(tmp_path):
    retriever = Retriever(
        config=make_config(tmp_path, threshold=0.95), embed_fn=lambda _: [1, 1]
    )
    _, rejected = retriever.retrieve("query")
    assert rejected


def test_empty_input(tmp_path):
    retriever = Retriever(config=make_config(tmp_path), embed_fn=lambda _: [1, 0])
    with pytest.raises(ValueError, match="nhập câu hỏi"):
        retriever.retrieve(" ")


def test_metadata_mismatch(tmp_path):
    config = make_config(tmp_path)
    config.metadata_path.write_text(
        json.dumps({"faqs": [], "content_hash": "x"}), encoding="utf-8"
    )
    with pytest.raises(StorageError, match="không khớp"):
        Retriever(config=config, embed_fn=lambda _: [1, 0])
