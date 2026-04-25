"""Captions-first transcript fetching, with a Whisper fallback.

Returns chunks whose size is chosen for citation precision — the LLM sees
the full transcript as one stream, but each chunk gets a [MM:SS] label
that becomes the target for clickable timestamps in the UI. Both the
caption path and the Whisper path run through the same joiner so the
chunk shape is identical regardless of source.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from openai import OpenAI
from pydantic import BaseModel
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)
from youtube_transcript_api.proxies import GenericProxyConfig

from app.config import OPENAI_API_KEY, PROXY_URL

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
    """Fetch captions for a video and group them into ~15s chunks."""
    api = YouTubeTranscriptApi(proxy_config=_proxy_config()) if PROXY_URL else YouTubeTranscriptApi()
    try:
        snippets = api.fetch(video_id=video_id)
    except (NoTranscriptFound, TranscriptsDisabled) as e:
        raise TranscriptUnavailable(str(e)) from e

    return _join_snippets(snippets)


def _proxy_config() -> GenericProxyConfig:
    # Same URL for both schemes — Oxylabs and most residential proxies accept
    # CONNECT for HTTPS over the same HTTP endpoint.
    return GenericProxyConfig(http_url=PROXY_URL, https_url=PROXY_URL)


def transcribe_audio(video_id: str) -> list[Chunk]:
    """Fallback path: pull audio with yt-dlp, transcribe via OpenAI Whisper.

    Runs synchronously and can take tens of seconds for long videos. The
    /api/load route caches the result so only the first viewer pays.
    """
    if not OPENAI_API_KEY:
        raise TranscriptUnavailable("OPENAI_API_KEY not set; Whisper fallback disabled")

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / f"{video_id}.m4a"
        _download_audio(video_id, audio_path)
        segments = _whisper_transcribe(audio_path)

    # Whisper's segment shape is (start, end, text); adapt to the snippet
    # shape the joiner expects so both transcript paths share one code path.
    # Older openai SDK versions return list[dict] here, newer ones return
    # pydantic objects, so support both via _seg.
    adapted = [
        SimpleNamespace(
            text=_seg(s, "text").strip(),
            start=_seg(s, "start"),
            duration=_seg(s, "end") - _seg(s, "start"),
        )
        for s in segments
    ]
    return _join_snippets(adapted)


def _seg(segment, key):
    return segment[key] if isinstance(segment, dict) else getattr(segment, key)


def _download_audio(video_id: str, out_path: Path) -> None:
    # -f bestaudio[ext=m4a]/bestaudio keeps it audio-only and avoids re-
    # encoding when possible, which is both faster and cheaper to ship to
    # Whisper (m4a is well under the 25MB upload limit for typical videos).
    cmd = [
        "yt-dlp",
        "-f", "bestaudio[ext=m4a]/bestaudio",
        "-o", str(out_path),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    if PROXY_URL:
        # Same proxy as the captions path; YouTube blocks AWS IPs hard
        # otherwise.
        cmd.extend(["--proxy", PROXY_URL])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise TranscriptUnavailable(f"yt-dlp failed: {result.stderr[:200]}")


def _whisper_transcribe(audio_path: Path):
    # verbose_json already returns segment-level start/end/text; word-level
    # timestamps would require the (newer-SDK) timestamp_granularities flag,
    # but segment is precise enough for our citation chunking.
    client = OpenAI(api_key=OPENAI_API_KEY)
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )
    return response.segments


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
