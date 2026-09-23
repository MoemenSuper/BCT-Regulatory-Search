# syntax=docker/dockerfile:1
# Pilot image: FastAPI + React UI + baked PDF corpus and Voyage search indexes.

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

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "google-genai>=2.20.0,<3" "filelock>=3.18,<4"

COPY backend/ ./
COPY --from=ui /ui/dist /app/static
# Bake the local documents/ tree into the image (~100MB). Requires documents/ at build time
# (gitignored — present on the builder machine). Recipients get PDFs from the image, no host mount.
COPY documents/ /data/documents/
# Slim Voyage runtime assets (chunks + indexes). Seeded into the assets volume on first boot.
# Build with: python backend/tmp/export_baked_assets.py --source <runtime-assets-root>
COPY baked-runtime-assets/ /opt/bct/baked-assets/

ENV BCT_STATIC_DIR=/app/static \
    BCT_ASSETS_DIR=/data/assets \
    BCT_BAKED_ASSETS_DIR=/opt/bct/baked-assets \
    BCT_DOCUMENTS_DIR=/data/documents \
    BCT_DATA_DIR=/data/state \
    BCT_BIND_HOST=0.0.0.0 \
    BCT_BIND_PORT=8000 \
    BCT_INGEST_LOCAL_INDEX=0 \
    BCT_DEFAULT_PROFILE=cloud \
    BCT_CLOUD_RETRIEVAL_PROVIDER=voyage \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "docker_serve.py"]
