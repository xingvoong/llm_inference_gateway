"""
Tests for response caching and batch endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from app import cache
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_cache():
    cache.clear()
    yield
    cache.clear()


# --- cache tests ---

def test_cache_miss_returns_none():
    assert cache.get("some prompt", "gpt-4") is None


def test_cache_set_and_get():
    cache.set("hello", "gpt-4", "world")
    assert cache.get("hello", "gpt-4") == "world"


def test_cache_different_model_is_separate():
    cache.set("hello", "gpt-4", "response-a")
    assert cache.get("hello", "mistral") is None


def test_cache_hit_returns_same_response():
    # First call populates the cache
    r1 = client.post("/chat", json={"prompt": "cache me", "priority": "high"})
    assert r1.status_code == 200
    first_response = r1.json()["response"]

    # Second call should hit cache and return identical response
    r2 = client.post("/chat", json={"prompt": "cache me", "priority": "high"})
    assert r2.status_code == 200
    assert r2.json()["response"] == first_response


# --- batch endpoint tests ---

def test_batch_returns_200():
    response = client.post("/chat/batch", json={"prompts": ["hello", "world"]})
    assert response.status_code == 200


def test_batch_returns_correct_count():
    prompts = ["one", "two", "three"]
    response = client.post("/chat/batch", json={"prompts": prompts})
    data = response.json()
    assert len(data["results"]) == 3


def test_batch_each_result_has_required_fields():
    response = client.post("/chat/batch", json={"prompts": ["a", "b"]})
    for result in response.json()["results"]:
        assert "response" in result
        assert "model_used" in result


def test_batch_priority_applies_to_all():
    response = client.post(
        "/chat/batch",
        json={"prompts": ["first", "second"], "priority": "high"},
    )
    for result in response.json()["results"]:
        assert "gpt-4" in result["model_used"]


def test_batch_empty_prompts_returns_empty():
    response = client.post("/chat/batch", json={"prompts": []})
    assert response.status_code == 200
    assert response.json()["results"] == []
