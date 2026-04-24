"""Settings loaded from environment variables.

python-dotenv reads a local .env file in development; Docker and prod
inject env vars directly, so load_dotenv is a no-op there.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# Empty-string default so the app can start without a key (health checks,
# /api/load, etc. keep working). The Anthropic SDK raises a clear 401 on
# the first real call if the key is missing.
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
