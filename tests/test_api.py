"""
Integration tests for the /chat endpoint.

Tests the full request/response cycle using FastAPI's test client.
No real API calls — providers are mocked.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

client = TestClient(app)


def test_chat_returns_200():
    response = client.post("/chat", json={"prompt": "hello"})
    assert response.status_code == 200


def test_chat_response_has_required_fields():
    response = client.post("/chat", json={"prompt": "hello"})
    data = response.json()
    assert "response" in data
    assert "model_used" in data


def test_high_priority_uses_gpt4():
    response = client.post("/chat", json={"prompt": "explain something", "priority": "high"})
    assert response.status_code == 200
    assert "gpt-4" in response.json()["model_used"]


def test_low_cost_uses_fast_model():
    response = client.post("/chat", json={"prompt": "explain something", "max_cost": 0.005})
    assert response.status_code == 200
    assert "Mistral" in response.json()["model_used"]


def test_invalid_priority_returns_422():
    response = client.post("/chat", json={"prompt": "hello", "priority": "urgent"})
    assert response.status_code == 422


def test_missing_prompt_returns_422():
    response = client.post("/chat", json={"priority": "high"})
    assert response.status_code == 422


def test_empty_prompt_returns_200():
    response = client.post("/chat", json={"prompt": ""})
    assert response.status_code == 200


def test_request_is_logged(tmp_path, monkeypatch):
    import app.logger as logger_module
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(logger_module, "DB_PATH", db_path)
    logger_module.init_db()

    client.post("/chat", json={"prompt": "test logging", "priority": "high"})

    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM requests").fetchall()
    conn.close()
    assert len(rows) == 1
    assert "test logging" in rows[0][2]  # prompt field
