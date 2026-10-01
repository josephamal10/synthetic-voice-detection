"""User accounts: sign up, sign in, and security-question password reset.

Kept separate from app.py so the detection pipeline stays untouched — this
module owns its own SQLite database (users.db) and never imports Streamlit
itself, so it can be unit-tested or reused without a running app.
"""
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager

import bcrypt

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "analysis_audio")

SECURITY_QUESTIONS = [
    "What was the name of your first pet?",
    "What is your mother's maiden name?",
    "What city were you born in?",
    "What was the name of your first school?",
    "What is your favourite teacher's name?",
]


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                phone_number TEXT,
                security_question TEXT NOT NULL,
                security_answer_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                score TEXT NOT NULL,
                threshold TEXT NOT NULL,
                model TEXT NOT NULL,
                margin TEXT,
                audio_path TEXT,
                analysed_at TEXT NOT NULL
            )
            """
        )
        # Migrate a table created before margin/audio_path existed (a plain
        # CREATE TABLE IF NOT EXISTS above won't add columns to it).
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(analysis_records)")}
        if "margin" not in existing_cols:
            conn.execute("ALTER TABLE analysis_records ADD COLUMN margin TEXT")
        if "audio_path" not in existing_cols:
            conn.execute("ALTER TABLE analysis_records ADD COLUMN audio_path TEXT")


# ---------------------------------------------------------------- passwords
def validate_password_strength(password):
    """Returns (is_valid, message). Enforces alphanumeric + strong:
    at least 8 characters, both a letter and a digit, and both an
    uppercase and a lowercase letter."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return False, "Password must be alphanumeric — it needs at least one letter and one number."
    if not re.search(r"[a-z]", password):
        return False, "Password must include at least one lowercase letter."
    if not re.search(r"[A-Z]", password):
        return False, "Password must include at least one uppercase letter."
    return True, ""


def password_strength_label(password):
    """Live-feedback label for the signup form, not a validation gate."""
    if not password:
        return "", ""
    score = sum([
        len(password) >= 8,
        len(password) >= 12,
        bool(re.search(r"[a-z]", password)) and bool(re.search(r"[A-Z]", password)),
        bool(re.search(r"\d", password)),
        bool(re.search(r"[^A-Za-z0-9]", password)),
    ])
    if score <= 2:
        return "Weak", "synthetic"
    if score <= 3:
        return "Medium", "warn"
    return "Strong", "genuine"


def _hash(text):
    return bcrypt.hashpw(text.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _check(text, hashed):
    try:
        return bcrypt.checkpw(text.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


# ---------------------------------------------------------------- accounts
def username_exists(username):
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        return row is not None


def create_user(first_name, last_name, username, password, phone_number, security_question, security_answer):
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (first_name, last_name, username, password_hash,
                                phone_number, security_question, security_answer_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                first_name.strip(),
                last_name.strip(),
                username.strip(),
                _hash(password),
                phone_number.strip() or None,
                security_question,
                _hash(security_answer.strip().lower()),
            ),
        )


def verify_login(username, password):
    """Returns the user row (as a dict) on success, or None."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
    if row is None:
        return None
    if not _check(password, row["password_hash"]):
        return None
    return dict(row)


def get_security_question(username):
    with _connect() as conn:
        row = conn.execute(
            "SELECT security_question FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
    return row["security_question"] if row else None


def verify_security_answer(username, answer):
    with _connect() as conn:
        row = conn.execute(
            "SELECT security_answer_hash FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
    if row is None:
        return False
    return _check(answer.strip().lower(), row["security_answer_hash"])


def reset_password(username, new_password):
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (_hash(new_password), username.strip()),
        )


# ---------------------------------------------------------------- analysis history
# Persisted per account (not per browser session) so the Session Report and
# the sidebar's ANALYSED count survive a logout/refresh/server restart —
# each user only ever sees their own clips, never another account's. The
# original audio is saved to disk (not as a DB blob) so a past entry can
# later be reopened for full playback and re-plotted signal analysis, not
# just its recorded score — see get_audio_bytes / page_report's detail view.
def add_analysis_record(user_id, filename, prediction, score, threshold, model,
                         margin, analysed_at, audio_bytes=None):
    audio_path = None
    if audio_bytes:
        user_dir = os.path.join(AUDIO_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        ext = os.path.splitext(filename)[1] or ".audio"
        audio_path = f"{user_id}/{uuid.uuid4().hex}{ext}"
        with open(os.path.join(AUDIO_DIR, audio_path), "wb") as f:
            f.write(audio_bytes)

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO analysis_records
                (user_id, filename, prediction, score, threshold, model, margin, audio_path, analysed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, filename, prediction, score, threshold, model, margin, audio_path, analysed_at),
        )


def get_analysis_records(user_id):
    """Oldest first, matching the previous in-session log's append order."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, prediction, score, threshold, model, margin, audio_path, analysed_at
            FROM analysis_records WHERE user_id = ? ORDER BY id ASC
            """,
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_audio_bytes(audio_path):
    """None if this record predates audio storage, or its file was removed."""
    if not audio_path:
        return None
    full_path = os.path.join(AUDIO_DIR, audio_path)
    if not os.path.exists(full_path):
        return None
    with open(full_path, "rb") as f:
        return f.read()


def count_analysis_records(user_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM analysis_records WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row["c"] if row else 0


def _remove_audio_file(audio_path):
    if not audio_path:
        return
    full_path = os.path.join(AUDIO_DIR, audio_path)
    if os.path.exists(full_path):
        try:
            os.remove(full_path)
        except OSError:
            pass  # best-effort cleanup — a stray file is harmless


def delete_analysis_record(user_id, record_id):
    """Scoped by user_id too, so one account can never delete another's clip."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT audio_path FROM analysis_records WHERE id = ? AND user_id = ?",
            (record_id, user_id),
        ).fetchone()
        conn.execute("DELETE FROM analysis_records WHERE id = ? AND user_id = ?", (record_id, user_id))
    if row:
        _remove_audio_file(row["audio_path"])


def clear_analysis_records(user_id):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT audio_path FROM analysis_records WHERE user_id = ?", (user_id,)
        ).fetchall()
        conn.execute("DELETE FROM analysis_records WHERE user_id = ?", (user_id,))
    for row in rows:
        _remove_audio_file(row["audio_path"])
