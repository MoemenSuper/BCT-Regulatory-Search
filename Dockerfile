# syntax=docker/dockerfile:1
# Pilot image: FastAPI + React UI. Mount PDFs; ingest builds search indexes.
# Graph Lite Neo4j is started by docker-compose.yml (not this Dockerfile alone).

FROM node:22-bookworm AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements-graph.txt ./
RUN pip install --no-cache-dir -r requirements-graph.txt \
    && pip install --no-cache-dir "google-genai>=2.20.0,<3" "filelock>=3.18,<4"

COPY backend/ ./
COPY --from=ui /ui/dist /app/static

ENV BCT_STATIC_DIR=/app/static \
    BCT_ASSETS_DIR=/data/assets \
    BCT_DOCUMENTS_DIR=/data/documents \
    BCT_DATA_DIR=/data/state \
    BCT_BIND_HOST=0.0.0.0 \
    BCT_BIND_PORT=8000 \
    BCT_ENABLE_GRAPH=1 \
    BCT_INGEST_GRAPH=1 \
    BCT_INGEST_LOCAL_INDEX=0 \
    BCT_DEFAULT_PROFILE=cloud \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "docker_serve.py"]
