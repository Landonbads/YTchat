"""Claude Opus 4.7 wrapper for chat (and, in commit #5, summary).

Prompt caching is applied to the transcript block so that follow-up
questions in the same conversation only pay full price for the first
request in a 5-minute window; subsequent requests hit Anthropic's
ephemeral cache at roughly a tenth of the cost and respond faster.
"""
from anthropic import Anthropic
from pydantic import BaseModel

from app.config import ANTHROPIC_API_KEY
from app.timestamps import seconds_to_hhmmss
from app.transcripts import Chunk

MODEL = "claude-opus-4-7"
MAX_CHAT_TOKENS = 2048
MAX_SUMMARY_TOKENS = 8192

_client = Anthropic(api_key=ANTHROPIC_API_KEY)

_CHAT_INSTRUCTIONS = (
    "You are a helpful assistant that answers questions about a YouTube "
    "video. The full transcript is provided below this instruction, with "
    "each segment labeled by its timestamp. When referencing specific "
    "moments in the video, cite them in [HH:MM:SS] or [MM:SS] format — "
    "the UI turns those into clickable links that seek the player. Be "
    "concise and direct; do not summarize the whole video unless asked."
)

_SUMMARY_INSTRUCTIONS = (
    "You are summarizing a YouTube video. The full transcript is provided "
    "below this instruction, with each segment labeled by its timestamp. "
    "Produce a clear summary covering the main points, structure, and "
    "notable moments. Cite [HH:MM:SS] or [MM:SS] timestamps for key "
    "moments so the reader can jump to them in the embedded player. Use "
    "short paragraphs and bullet points where helpful. Be information-"
    "dense; no filler."
)


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


def chat(chunks: list[Chunk], messages: list[Message]) -> str:
    """Send a chat turn to Claude with the transcript cached as context."""
    transcript = _format_transcript(chunks)

    response = _client.messages.create(
        model=MODEL,
        max_tokens=MAX_CHAT_TOKENS,
        system=[
            {"type": "text", "text": _CHAT_INSTRUCTIONS},
            # cache_control on the largest stable block — the transcript.
            # Same video + any user turns → same cached prefix → cheap follow-ups.
            {
                "type": "text",
                "text": f"Transcript:\n{transcript}",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[m.model_dump() for m in messages],
    )

    return response.content[0].text


def summarize(chunks: list[Chunk]) -> str:
    """One-shot summary of the video. Summary and chat use separate caches."""
    transcript = _format_transcript(chunks)

    response = _client.messages.create(
        model=MODEL,
        max_tokens=MAX_SUMMARY_TOKENS,
        system=[
            {"type": "text", "text": _SUMMARY_INSTRUCTIONS},
            {
                "type": "text",
                "text": f"Transcript:\n{transcript}",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[{"role": "user", "content": "Summarize this video."}],
    )

    return response.content[0].text


def _format_transcript(chunks: list[Chunk]) -> str:
    return "\n".join(
        f"[{seconds_to_hhmmss(c.start)}] {c.text}" for c in chunks
    )
