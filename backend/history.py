"""SQLite-backed chat history."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chat_history.db"


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                refused INTEGER NOT NULL DEFAULT 0,
                sources_json TEXT NOT NULL DEFAULT '[]',
                metrics_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.commit()


def save_entry(question: str, answer: str, refused: bool, sources: list, metrics: dict) -> int:
    init_db()
    with _conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO chat_history (created_at, question, answer, refused, sources_json, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                question,
                answer,
                1 if refused else 0,
                json.dumps(sources),
                json.dumps(metrics),
            ),
        )
        conn.commit()
        return cur.lastrowid


def list_history(limit: int = 50) -> list[dict]:
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_entry(entry_id: int) -> dict | None:
    init_db()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM chat_history WHERE id = ?", (entry_id,)).fetchone()
    return _row_to_dict(row) if row else None


def clear_history():
    init_db()
    with _conn() as conn:
        conn.execute("DELETE FROM chat_history")
        conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "question": row["question"],
        "answer": row["answer"],
        "refused": bool(row["refused"]),
        "sources": json.loads(row["sources_json"]),
        "metrics": json.loads(row["metrics_json"]),
    }
