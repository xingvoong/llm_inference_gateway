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
            routing_reason TEXT,
            latency_ms REAL,
            response_length INTEGER
        )
    """)
    # migrate existing databases that don't have routing_reason column
    existing = [row[1] for row in conn.execute("PRAGMA table_info(requests)")]
    if "routing_reason" not in existing:
        conn.execute("ALTER TABLE requests ADD COLUMN routing_reason TEXT")
    conn.commit()
    conn.close()


def log_request(prompt: str, selected_model: str, routing_reason: str, latency_ms: float, response_length: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO requests (timestamp, prompt, selected_model, routing_reason, latency_ms, response_length) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), prompt, selected_model, routing_reason, latency_ms, response_length)
    )
    conn.commit()
    conn.close()
