import sqlite3

DB_PATH = "logs/requests.db"


def compute_metrics():
    conn = sqlite3.connect(DB_PATH)

    print("=== Usage per model ===")
    for row in conn.execute("SELECT selected_model, COUNT(*) FROM requests GROUP BY selected_model"):
        print(f"  {row[0]}: {row[1]} requests")

    print("\n=== Avg latency per model (ms) ===")
    for row in conn.execute("SELECT selected_model, ROUND(AVG(latency_ms), 2) FROM requests GROUP BY selected_model"):
        print(f"  {row[0]}: {row[1]}ms")

    conn.close()


if __name__ == "__main__":
    compute_metrics()
