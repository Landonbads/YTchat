"""Settings loaded from environment variables.

python-dotenv reads a local .env file in development; Docker and prod
inject env vars directly, so load_dotenv is a no-op there.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Empty-string defaults so the app boots without keys. The Anthropic and
# OpenAI clients raise clear auth errors on the first real call.
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# SQLite cache lives in a mounted Docker volume in prod so transcripts
# survive container restarts; local dev defaults to a file next to the repo.
DB_PATH: Path = Path(os.environ.get("YTCHAT_DB_PATH", "./ytchat.db"))

# Optional HTTP proxy for both youtube-transcript-api and yt-dlp. AWS IPs
# get rate-limited or blocked by YouTube under sustained traffic, so we
# route YouTube-bound requests through a residential proxy when set. Empty
# string means "no proxy"; format is http://user:pass@host:port.
PROXY_URL: str = os.environ.get("PROXY_URL", "")
