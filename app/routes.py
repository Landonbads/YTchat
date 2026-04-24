"""All HTTP endpoints.

Split into two routers so main.py stays pure wiring:

* root_router — infrastructure (serve the HTML page, health check)
* api_router  — product API under /api/*
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import llm
from app.transcripts import (
    Chunk,
    TranscriptUnavailable,
    extract_video_id,
    get_transcript,
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


class LoadResponse(BaseModel):
    video_id: str
    chunks: list[Chunk]


@api_router.post("/load", response_model=LoadResponse)
def load(req: LoadRequest) -> LoadResponse:
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        chunks = get_transcript(video_id)
    except TranscriptUnavailable as e:
        # Commit #9 will catch this and transcribe via Whisper instead.
        raise HTTPException(status_code=422, detail=f"Transcript unavailable: {e}")

    return LoadResponse(video_id=video_id, chunks=chunks)


class ChatRequest(BaseModel):
    video_id: str
    messages: list[llm.Message]


class ChatResponse(BaseModel):
    content: str


@api_router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # Transcript is re-fetched on every chat turn for now; the SQLite cache
    # in commit #10 will make this free after the first load.
    try:
        chunks = get_transcript(req.video_id)
    except TranscriptUnavailable as e:
        raise HTTPException(status_code=422, detail=f"Transcript unavailable: {e}")

    reply = llm.chat(chunks, req.messages)
    return ChatResponse(content=reply)


class SummaryRequest(BaseModel):
    video_id: str


class SummaryResponse(BaseModel):
    content: str


@api_router.post("/summary", response_model=SummaryResponse)
def summary(req: SummaryRequest) -> SummaryResponse:
    try:
        chunks = get_transcript(req.video_id)
    except TranscriptUnavailable as e:
        raise HTTPException(status_code=422, detail=f"Transcript unavailable: {e}")

    return SummaryResponse(content=llm.summarize(chunks))
