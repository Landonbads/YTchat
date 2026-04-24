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
VOLUME ["/data"]

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
