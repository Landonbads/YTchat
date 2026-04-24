"""Captions-first transcript fetching.

Returns chunks whose size is chosen for citation precision — the LLM sees
the full transcript as one stream, but each chunk gets a [MM:SS] label
that becomes the target for clickable timestamps in the UI. Whisper
fallback for videos without captions lands in commit #9.
"""
from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

# Chunk size trades citation precision against label overhead: smaller
# chunks mean clicking a [MM:SS] citation lands closer to the moment the
# model was referencing, at the cost of more timestamp labels in the
# transcript. At 15s the token cost of the extra labels is negligible and
# citations land within a scrub or two of the actual content.
CHUNK_SECONDS = 15.0

# YouTube video IDs are always exactly 11 chars from this alphabet.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class TranscriptUnavailable(Exception):
    """Captions could not be fetched. Commit #9 catches this and tries Whisper."""


class Chunk(BaseModel):
    start: float      # seconds from video start — used by player.seekTo() on the frontend
    duration: float
    text: str


def extract_video_id(url_or_id: str) -> str:
    """Accept a full YouTube URL or a bare ID; return the 11-char video ID."""
    s = url_or_id.strip()
    if _VIDEO_ID_RE.match(s):
        return s

    parsed = urlparse(s)
    host = parsed.hostname or ""
    if host.endswith("youtu.be"):
        candidate = parsed.path.lstrip("/")
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            candidate = (parse_qs(parsed.query).get("v") or [""])[0]
        elif parsed.path.startswith(("/shorts/", "/embed/")):
            candidate = parsed.path.split("/")[2]
        else:
            candidate = ""
    else:
        candidate = ""

    if not _VIDEO_ID_RE.match(candidate):
        raise ValueError(f"Could not extract a YouTube video ID from: {url_or_id!r}")
    return candidate


def get_transcript(video_id: str) -> list[Chunk]:
    """Fetch captions for a video and group them into ~30s chunks."""
    try:
        snippets = YouTubeTranscriptApi().fetch(video_id=video_id)
    except (NoTranscriptFound, TranscriptsDisabled) as e:
        raise TranscriptUnavailable(str(e)) from e

    return _join_snippets(snippets)


def _join_snippets(snippets) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur_text = ""
    cur_start = 0.0
    cur_duration = 0.0

    for snip in snippets:
        if cur_text and cur_duration + snip.duration > CHUNK_SECONDS:
            chunks.append(Chunk(start=cur_start, duration=cur_duration, text=cur_text))
            cur_text = snip.text
            cur_start = snip.start
            cur_duration = snip.duration
        else:
            if not cur_text:
                cur_start = snip.start
                cur_text = snip.text
            else:
                cur_text += " " + snip.text
            cur_duration += snip.duration

    if cur_text:
        chunks.append(Chunk(start=cur_start, duration=cur_duration, text=cur_text))

    return chunks
