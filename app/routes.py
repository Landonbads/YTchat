"""All HTTP endpoints.

Split into two routers so main.py stays pure wiring:

* root_router — infrastructure (serve the HTML page, health check)
* api_router  — product API under /api/*
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import cache, llm
from app.transcripts import (
    Chunk,
    TranscriptUnavailable,
    extract_video_id,
    get_transcript,
    transcribe_audio,
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

root_router = APIRouter()
api_router = APIRouter(prefix="/api")


@root_router.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@root_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class LoadRequest(BaseModel):
    url: str
    # Whisper runs tens of seconds per video, so the frontend opts in on a
    # second call after showing the user a "generating transcript…" status.
    allow_whisper: bool = False


class LoadResponse(BaseModel):
    video_id: str
    chunks: list[Chunk]


@api_router.post("/load", response_model=LoadResponse)
def load(req: LoadRequest) -> LoadResponse:
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cached = cache.get(video_id)
    if cached is not None:
        return LoadResponse(video_id=video_id, chunks=cached)

    try:
        chunks = get_transcript(video_id)
        source = "captions"
    except TranscriptUnavailable as e:
        if not req.allow_whisper:
            # Frontend sees 422, shows a "Generating with Whisper…" status,
            # and retries with allow_whisper=true.
            raise HTTPException(
                status_code=422,
                detail=f"Captions unavailable ({e}); retry with allow_whisper=true",
            )
        try:
            chunks = transcribe_audio(video_id)
            source = "whisper"
        except TranscriptUnavailable as we:
            raise HTTPException(status_code=422, detail=f"Whisper failed: {we}")

    cache.put(video_id, chunks, source=source)
    return LoadResponse(video_id=video_id, chunks=chunks)


def _require_cached(video_id: str) -> list[Chunk]:
    chunks = cache.get(video_id)
    if chunks is None:
        raise HTTPException(status_code=400, detail="Video not loaded; POST /api/load first")
    return chunks


# Chat history is held client-side: the browser keeps the conversation in
# memory and sends the full messages array on every /api/chat call. The
# server stays stateless, which makes multi-user safety automatic — two
# browsers can never see each other's chat. The cap below just prevents a
# single client from blasting an unbounded history at Claude.
MAX_HISTORY_MESSAGES = 50


class ChatRequest(BaseModel):
    video_id: str
    messages: list[llm.Message]


class ChatResponse(BaseModel):
    content: str


@api_router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if len(req.messages) > MAX_HISTORY_MESSAGES:
        raise HTTPException(
            status_code=413,
            detail=f"Chat history too long ({len(req.messages)} > {MAX_HISTORY_MESSAGES})",
        )
    chunks = _require_cached(req.video_id)
    return ChatResponse(content=llm.chat(chunks, req.messages))


class SummaryRequest(BaseModel):
    video_id: str


class SummaryResponse(BaseModel):
    content: str


@api_router.post("/summary", response_model=SummaryResponse)
def summary(req: SummaryRequest) -> SummaryResponse:
    chunks = _require_cached(req.video_id)
    return SummaryResponse(content=llm.summarize(chunks))
