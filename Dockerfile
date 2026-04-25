FROM python:3.12-slim

# ffmpeg is needed by yt-dlp to extract/convert audio for the Whisper fallback.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so the pip layer stays cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/

# Persistent transcript cache lives on a mounted volume so it survives
# container restarts and is shared across viewers of the same video.
ENV YTCHAT_DB_PATH=/data/ytchat.db
RUN mkdir -p /data

# Drop root for the runtime: if the app is ever exploited, the attacker
# is stuck as an unprivileged user. chown happens before VOLUME so the
# ownership is baked into the volume metadata.
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app /data
USER app

VOLUME ["/data"]
EXPOSE 8000

# Lets `docker ps` (and any orchestrator we move to later) tell whether
# uvicorn is actually responding, not just whether the process is alive.
# Uses python so we don't need to install curl in the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
