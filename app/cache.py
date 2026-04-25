"""Persistent transcript cache keyed by video ID.

The cache is shared across users — fetching or transcribing a video is
expensive, but the result is the same for everyone, so the first viewer
pays and everyone after is instant. Per-user state (chat history) lives
in the browser, not here.
"""
import json
import sqlite3
import time
from threading import Lock

from app.config import DB_PATH
from app.transcripts import Chunk

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# A single connection with check_same_thread=False plus an explicit mutex
# is simpler than a connection pool and plenty for our write volume
# (one insert per cold-cache video load).
_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
_lock = Lock()

_conn.executescript(
    """
    CREATE TABLE IF NOT EXISTS transcripts (
        video_id    TEXT PRIMARY KEY,
        source      TEXT NOT NULL,   -- 'captions' or 'whisper'
        chunks_json TEXT NOT NULL,
        fetched_at  REAL NOT NULL
    );
    """
)


def get(video_id: str) -> list[Chunk] | None:
    with _lock:
        row = _conn.execute(
            "SELECT chunks_json FROM transcripts WHERE video_id = ?",
            (video_id,),
        ).fetchone()
    if not row:
        return None
    return [Chunk.model_validate(c) for c in json.loads(row[0])]


def put(video_id: str, chunks: list[Chunk], source: str) -> None:
    payload = json.dumps([c.model_dump() for c in chunks])
    with _lock:
        _conn.execute(
            "INSERT OR REPLACE INTO transcripts "
            "(video_id, source, chunks_json, fetched_at) VALUES (?, ?, ?, ?)",
            (video_id, source, payload, time.time()),
        )
        _conn.commit()
