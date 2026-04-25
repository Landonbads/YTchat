# YTchat

Chat with any YouTube video. Paste a URL, get an instant summary, ask questions,
and click any `[MM:SS]` citation to jump the embedded player to that moment.

Live at **[ytchat.us](https://ytchat.us)**.

## How it works

1. You paste a URL. YTchat fetches captions via `youtube-transcript-api`.
   If captions are disabled, it falls back to `yt-dlp` + OpenAI Whisper.
2. The full transcript is sent to Claude Opus 4.7 as a **prompt-cached**
   system message. Follow-up questions reuse the cache and are cheap.
3. Answers cite timestamps. Clicks seek the embedded player — no new tabs.
4. Transcripts are cached in SQLite by video ID, so the next viewer gets
   the video instantly.

## Stack

- **Backend:** FastAPI (async, single-process, multi-user safe)
- **Frontend:** vanilla HTML + JS. No framework, no build step. The
  YouTube IFrame API drives the embedded player.
- **LLM:** Anthropic Claude Opus 4.7 (`claude-opus-4-7`) with prompt caching
- **Transcription fallback:** yt-dlp + OpenAI Whisper API
- **Persistence:** SQLite (transcripts keyed by video ID)
- **Deploy:** Docker container on AWS

No RAG in the single-video path — in 2026, the full transcript fits in
context and cached tokens beat retrieval on quality and latency. A
`channel/playlist` mode (where RAG pays off) is scaffolded for later.

## Run locally

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000.

## Run in Docker

```bash
docker build -t ytchat .
docker run -p 8000:8000 --env-file .env -v ytchat-cache:/data ytchat
```

The volume persists the SQLite transcript cache across restarts.

## Deploy on AWS EC2

The production image runs on a single EC2 instance — the same host
serving ytchat.us. Build, then run with `--restart unless-stopped` so the
container comes back automatically after a reboot or crash:

```bash
docker build -t ytchat .
docker run -d \
  --name ytchat \
  --restart unless-stopped \
  -p 80:8000 \
  --env-file .env \
  -v ytchat-cache:/data \
  ytchat
```

The `HEALTHCHECK` in the Dockerfile makes `docker ps` report
`healthy`/`unhealthy` based on whether `/health` is actually responding,
which is useful both for manual debugging and for any orchestrator (ECS,
Nomad) you might adopt later without changing the image.

## Environment

| Var                 | Required            | Purpose                          |
|---------------------|---------------------|----------------------------------|
| `ANTHROPIC_API_KEY` | yes                 | Claude Opus 4.7 for chat/summary |
| `OPENAI_API_KEY`    | only if no captions | Whisper fallback transcription   |

## Project layout

```
app/          FastAPI backend, one concern per module
static/       vanilla HTML/JS/CSS served by FastAPI
tests/        unit tests for transcript parsing & timestamp logic
```

## Status

Rewritten April 2026. The original December 2023 implementation (Gradio +
ChromaDB + GPT-3.5) is preserved in git history prior to this commit.
