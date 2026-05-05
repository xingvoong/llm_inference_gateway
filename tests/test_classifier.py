"""
Unit tests for the zero-shot classifier and learned router.
"""

import pytest
from unittest.mock import patch, MagicMock


def test_classify_prompt_returns_valid_label():
    mock_result = {"labels": ["code generation", "summarization"], "scores": [0.8, 0.2]}
    with patch("app.classifier.get_classifier") as mock_get:
        mock_get.return_value = MagicMock(return_value=mock_result)
        from app.classifier import classify_prompt
        result = classify_prompt("write a function")
        assert result in ["code generation", "summarization", "question answering", "general chat"]


def test_learned_router_returns_string():
    from app.learned_router import predict_model, is_trained_model_available
    if not is_trained_model_available():
        pytest.skip("Trained model not available — run scripts/train_router.py first")
    result = predict_model("Write a Python function to sort a list")
    assert isinstance(result, str)
    assert len(result) > 0


def test_learned_router_code_prompt():
    from app.learned_router import predict_model, is_trained_model_available
    if not is_trained_model_available():
        pytest.skip("Trained model not available — run scripts/train_router.py first")
    result = predict_model("Write a Python function to sort a list")
    assert "Mistral" in result


def test_learned_router_general_prompt():
    from app.learned_router import predict_model, is_trained_model_available
    if not is_trained_model_available():
        pytest.skip("Trained model not available — run scripts/train_router.py first")
    result = predict_model("Why do stars twinkle at night?")
    assert "gpt" in result
