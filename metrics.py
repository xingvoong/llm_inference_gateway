import sqlite3
import os

DB_PATH = "logs/requests.db"


def compute_metrics():
    if not os.path.exists(DB_PATH):
        print("No logs found. Make some requests first.")
        return

    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]

    if total == 0:
        print("No requests logged yet.")
        conn.close()
        return

    print(f"=== Total requests: {total} ===\n")

    print("=== Usage per model ===")
    for row in conn.execute("SELECT selected_model, COUNT(*) FROM requests GROUP BY selected_model ORDER BY COUNT(*) DESC"):
        pct = round(row[1] / total * 100)
        print(f"  {row[0]}: {row[1]} requests ({pct}%)")

    print("\n=== Avg latency per model (ms) ===")
    for row in conn.execute("SELECT selected_model, ROUND(AVG(latency_ms), 2) FROM requests GROUP BY selected_model ORDER BY AVG(latency_ms)"):
        print(f"  {row[0]}: {row[1]}ms")

    print("\n=== Routing distribution ===")
    for row in conn.execute("SELECT routing_reason, COUNT(*) FROM requests GROUP BY routing_reason ORDER BY COUNT(*) DESC"):
        reason = row[0] or "unknown"
        pct = round(row[1] / total * 100)
        print(f"  {reason}: {row[1]} requests ({pct}%)")

    conn.close()


if __name__ == "__main__":
    compute_metrics()
