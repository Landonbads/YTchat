"""FastAPI entry point — wiring only. Every HTTP handler lives in app/routes.py."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import routes

app = FastAPI(title="YTchat")
app.include_router(routes.root_router)
app.include_router(routes.api_router)

# CSS/JS live under /static/*; the HTML page itself is served by
# routes.root_router so we return a specific file rather than exposing a
# directory listing.
app.mount("/static", StaticFiles(directory=routes.STATIC_DIR), name="static")
