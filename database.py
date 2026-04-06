"""
SQLite database setup for AI DJ user accounts and activity tracking.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "aidj.db")


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with row_factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # Migration: add is_admin to existing databases that predate this column
    existing_columns = [row[1] for row in cursor.execute("PRAGMA table_info(users)")]
    if "is_admin" not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        print("[DB] Migrated: added is_admin column to users")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            detail TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_songs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            song_name TEXT NOT NULL,
            song_data TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, song_name)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS setlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, name)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS setlist_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setlist_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            song_key TEXT NOT NULL,
            song_title TEXT NOT NULL,
            song_artist TEXT DEFAULT '',
            transition_exit_segment TEXT,
            transition_entry_segment TEXT,
            FOREIGN KEY (setlist_id) REFERENCES setlists(id) ON DELETE CASCADE,
            UNIQUE(setlist_id, position)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_user ON activity(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_type ON activity(event_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_songs_user ON user_songs(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_setlists_user ON setlists(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_setlist_items_setlist ON setlist_items(setlist_id)")

    conn.commit()
    conn.close()
    print("[DB] Database initialized")


def set_admin(email: str, is_admin: bool = True):
    """Grant or revoke admin privileges for a user by email.
    
    Usage from a Python shell or script:
        from database import set_admin
        set_admin('your@email.com')          # grant
        set_admin('your@email.com', False)   # revoke
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET is_admin = ? WHERE email = ?",
        (1 if is_admin else 0, email)
    )
    if cursor.rowcount == 0:
        print(f"[DB] No user found with email: {email}")
    else:
        status = "granted" if is_admin else "revoked"
        print(f"[DB] Admin {status} for {email}")
    conn.commit()
    conn.close()
