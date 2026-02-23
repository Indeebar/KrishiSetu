"""
Database setup — SQLite for local dev, PostgreSQL-compatible via DATABASE_URL env var.
"""

import os
import sqlite3
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_PATH = Path(__file__).parent.parent / "krishisetu.db"


def get_connection():
    """Returns a sqlite3 connection. Replace with psycopg2 for PostgreSQL."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they do not exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   DATETIME DEFAULT (datetime('now')),
            waste_type  TEXT NOT NULL,
            confidence  REAL NOT NULL,
            state       TEXT NOT NULL,
            price_per_kg REAL,
            image_name  TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialised.")


def insert_prediction(waste_type: str, confidence: float, state: str,
                      price_per_kg: float | None, image_name: str | None):
    conn = get_connection()
    conn.execute(
        """INSERT INTO predictions (waste_type, confidence, state, price_per_kg, image_name)
           VALUES (?, ?, ?, ?, ?)""",
        (waste_type, confidence, state, price_per_kg, image_name),
    )
    conn.commit()
    conn.close()


def fetch_history(limit: int = 20):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
