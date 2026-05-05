"""
Unit tests for the routing logic.

Tests each routing rule in isolation — no API calls, no model loading.
"""

from unittest.mock import patch
from app.router import route_request
from app.providers.openai_provider import OpenAIProvider
from app.providers.huggingface_provider import HuggingFaceProvider


def test_high_priority_routes_to_best_model():
    provider, model, reason = route_request(prompt="tell me something", priority="high")
    assert isinstance(provider, OpenAIProvider)
    assert "gpt" in model
    assert reason == "priority==high"


def test_low_cost_routes_to_fast_model():
    provider, model, reason = route_request(prompt="tell me something", max_cost=0.005)
    assert isinstance(provider, HuggingFaceProvider)
    assert reason == "max_cost<0.01"


def test_priority_overrides_cost():
    provider, model, reason = route_request(prompt="tell me something", priority="high", max_cost=0.001)
    assert isinstance(provider, OpenAIProvider)
    assert "gpt" in model
    assert reason == "priority==high"


def test_no_params_returns_default():
    prompt = "a" * 200
    with patch("app.router.is_trained_model_available", return_value=False):
        with patch("app.router.classify_prompt", return_value="general chat"):
            provider, model, reason = route_request(prompt=prompt)
            assert isinstance(provider, OpenAIProvider)
            assert "zero_shot" in reason


def test_code_prompt_routes_to_fast_model():
    prompt = "a" * 200
    with patch("app.router.is_trained_model_available", return_value=False):
        with patch("app.router.classify_prompt", return_value="code generation"):
            provider, model, reason = route_request(prompt=prompt)
            assert isinstance(provider, HuggingFaceProvider)
            assert reason == "zero_shot:code generation"


def test_summarization_routes_to_fast_model():
    prompt = "a" * 200
    with patch("app.router.is_trained_model_available", return_value=False):
        with patch("app.router.classify_prompt", return_value="summarization"):
            provider, model, reason = route_request(prompt=prompt)
            assert isinstance(provider, HuggingFaceProvider)
            assert reason == "zero_shot:summarization"


def test_learned_router_used_when_available():
    prompt = "Write a Python function"
    with patch("app.router.is_trained_model_available", return_value=True):
        with patch("app.router.predict_model", return_value="gpt-4") as mock_predict:
            provider, model, reason = route_request(prompt=prompt)
            mock_predict.assert_called_once_with(prompt)
            assert model == "gpt-4"
            assert reason == "learned_router"
