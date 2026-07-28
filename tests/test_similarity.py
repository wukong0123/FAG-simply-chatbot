import numpy as np
import pytest

from utils import cosine_scores, normalize_vector


def test_identical_vector_scores_one():
    assert cosine_scores(np.array([[1, 2]]), np.array([1, 2]))[0] == pytest.approx(1)


def test_orthogonal_vector_scores_zero():
    assert cosine_scores(np.array([[1, 0]]), np.array([0, 1]))[0] == pytest.approx(0)


@pytest.mark.parametrize("vector", [np.array([]), np.array([0, 0]), np.zeros((1, 2))])
def test_invalid_vector(vector):
    with pytest.raises(ValueError):
        normalize_vector(vector)


def test_shape_mismatch():
    with pytest.raises(ValueError, match="không khớp"):
        cosine_scores(np.ones((2, 3)), np.ones(2))
