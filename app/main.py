"""FastAPI entry point. Serves the static UI and a health check.

Routes are split into app/routes.py once there's more than the page itself
to serve — for now it's small enough to live inline.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="YTchat")

# Serve CSS/JS from /static/*; the HTML page is served separately at / so we
# can return it without exposing a directory listing.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
