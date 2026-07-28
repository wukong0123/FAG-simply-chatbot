import json

import pytest

from utils import DataValidationError, load_faqs, validate_faqs


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
