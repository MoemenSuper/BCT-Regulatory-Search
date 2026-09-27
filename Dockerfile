# syntax=docker/dockerfile:1
# Pilot image: FastAPI + React UI + baked PDF corpus, local indexes, and local ML stack.

FROM node:22-bookworm AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
WORKDIR /app

# OpenCV / EasyOCR / Paddle runtime libs
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libgomp1 \
        libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements-local.txt ./
# Full local_hybrid stack: retrieval (e5/Chroma/BGE) + visual ingest (EasyOCR + PaddleOCR-VL).
# Torch comes in as CPU via sentence-transformers / EasyOCR (fine for Docker Desktop).
RUN pip install --no-cache-dir -r requirements-local.txt \
    && pip install --no-cache-dir \
        "paddlepaddle==3.0.0" \
        -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
    && pip install --no-cache-dir "paddleocr[doc-parser]>=3.4,<4" \
    && pip install --no-cache-dir "google-genai>=2.20.0,<3" "filelock>=3.18,<4"

COPY backend/ ./
COPY --from=ui /ui/dist /app/static
# Bake the local documents/ tree into the image (~100MB). Requires documents/ at build time
# (gitignored — present on the builder machine). Recipients get PDFs from the image, no host mount.
COPY documents/ /data/documents/
# Slim local runtime assets (chunks + Chroma). Seeded into the assets volume on first boot.
# Build with: python backend/tmp/rebuild_local_baked.py
COPY baked-runtime-assets/ /opt/bct/baked-assets/

# CPU defaults for Docker Desktop; override with GPU image/flags if you have NVIDIA Container Toolkit.
ENV BCT_STATIC_DIR=/app/static \
    BCT_ASSETS_DIR=/data/assets \
    BCT_BAKED_ASSETS_DIR=/opt/bct/baked-assets \
    BCT_DOCUMENTS_DIR=/data/documents \
    BCT_DATA_DIR=/data/state \
    BCT_BIND_HOST=0.0.0.0 \
    BCT_BIND_PORT=8000 \
    BCT_INGEST_LOCAL_INDEX=1 \
    BCT_DEFAULT_PROFILE=local_hybrid \
    BCT_CLOUD_RETRIEVAL_PROVIDER=voyage \
    BCT_EASYOCR_GPU=0 \
    BCT_PADDLE_DEVICE=cpu \
    HF_HOME=/opt/bct/hf \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1 \
    PYTHONUNBUFFERED=1

# Pre-download embed / rerank / EasyOCR weights so first query / Arabic ingest is not a cold start.
RUN python -c "from embedding import create_embedding_model; create_embedding_model(); \
from reranker import create_reranker; create_reranker(); \
import easyocr; easyocr.Reader(['ar'], gpu=False, verbose=False); \
print('local models warmed')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "docker_serve.py"]
