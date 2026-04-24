FROM python:3.12-slim

WORKDIR /app

# Install deps first so pip layer stays cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
