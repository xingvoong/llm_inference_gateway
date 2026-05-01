import sqlite3
import os
from datetime import datetime

DB_PATH = "logs/requests.db"


def init_db():
    os.makedirs("logs", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            prompt TEXT,
            selected_model TEXT,
            latency_ms REAL,
            response_length INTEGER
        )
    """)
    conn.commit()
    conn.close()


def log_request(prompt: str, selected_model: str, latency_ms: float, response_length: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO requests (timestamp, prompt, selected_model, latency_ms, response_length) VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), prompt, selected_model, latency_ms, response_length)
    )
    conn.commit()
    conn.close()
