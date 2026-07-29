import json

import pytest

from utils import (
    DataValidationError,
    is_contextual_follow_up,
    load_faqs,
    normalize_retrieval_query,
    validate_faqs,
)


VALID = [{"id": 1, "question": " Câu hỏi? ", "answer": " Trả lời. "}]


def test_valid_faq_and_default_category():
    result = validate_faqs(VALID)
    assert result[0] == {
        "id": 1, "question": "Câu hỏi?", "answer": "Trả lời.", "category": "general"
    }


@pytest.mark.parametrize("field", ["question", "answer"])
def test_empty_required_text(field):
    faq = {**VALID[0], field: " "}
    with pytest.raises(DataValidationError, match=field):
        validate_faqs([faq])


def test_duplicate_id():
    with pytest.raises(DataValidationError, match="trùng"):
        validate_faqs(VALID * 2)


def test_valid_aliases_are_normalized():
    faq = {**VALID[0], "aliases": [" Cách hỏi khác? ", "Cách hỏi khác?"]}
    assert validate_faqs([faq])[0]["aliases"] == ["Cách hỏi khác?"]


def test_invalid_aliases():
    faq = {**VALID[0], "aliases": [""]}
    with pytest.raises(DataValidationError, match="aliases"):
        validate_faqs([faq])


def test_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{invalid", encoding="utf-8")
    with pytest.raises(DataValidationError, match="JSON không hợp lệ"):
        load_faqs(path)


def test_contextual_follow_up_detection():
    assert is_contextual_follow_up("Nó được diễn ra ở đâu?", has_history=True)
    assert not is_contextual_follow_up("Nó được diễn ra ở đâu?", has_history=False)
    assert not is_contextual_follow_up(
        "World Cup 2026 có bao nhiêu đội?", has_history=True
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("WC diễn ra ở đâu?", "FIFA World Cup 2026 diễn ra ở đâu?"),
        (
            "FIFA World Cup diễn ra khi nào?",
            "FIFA World Cup 2026 diễn ra khi nào?",
        ),
        (
            "World Cup 2026 có bao nhiêu đội?",
            "World Cup 2026 có bao nhiêu đội?",
        ),
    ],
)
def test_normalize_retrieval_query(question, expected):
    assert normalize_retrieval_query(question) == expected
