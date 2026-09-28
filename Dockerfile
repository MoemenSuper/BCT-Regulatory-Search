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
# Full local_hybrid stack: retrieval (e5/Chroma/BGE), EasyOCR and Docling. PyPI's Linux torch
# wheel includes CUDA: it runs on CPU, and on the GPU when the container is given one.
# Docling pulls the GUI OpenCV build; EasyOCR uses the headless one and both provide cv2, so the
# GUI build is removed and headless reinstalled.
RUN pip install --no-cache-dir -r requirements-local.txt \
    && pip install --no-cache-dir "google-genai>=2.20.0,<3" "filelock>=3.18,<4" "docling>=2.130,<3" \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps \
        "opencv-python-headless==$(python -c 'import importlib.metadata as m; print(m.version("opencv-python-headless"))')"

# PaddleOCR-VL (image regions, scanned French pages) in its own venv: its worker process is the only
# Paddle user, and keeping Paddle out of the torch environment rules out a clash of their GPU
# libraries. PADDLE=gpu builds the CUDA wheel (see docker-compose.gpu.yml); the default is CPU.
ARG PADDLE=cpu
RUN python -m venv /opt/paddle-venv \
    && if [ "$PADDLE" = "gpu" ]; then \
        /opt/paddle-venv/bin/pip install --no-cache-dir "paddlepaddle-gpu==3.2.1" \
            -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ ; \
    else \
        /opt/paddle-venv/bin/pip install --no-cache-dir "paddlepaddle==3.0.0" \
            -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ ; \
    fi \
    && /opt/paddle-venv/bin/pip install --no-cache-dir "paddleocr[doc-parser]>=3.4,<4"

COPY backend/ ./
COPY --from=ui /ui/dist /app/static
# Bake the local documents/ tree into the image (~100MB). Requires documents/ at build time
# (gitignored — present on the builder machine). Recipients get PDFs from the image, no host mount.
COPY documents/ /data/documents/
# Slim local runtime assets (chunks + Chroma). Seeded into the assets volume on first boot.
# Build with: python backend/tmp/rebuild_local_baked.py
COPY baked-runtime-assets/ /opt/bct/baked-assets/

# Devices are detected at run time (hardware.py; Paddle falls back to CPU by itself): the same image
# runs on a laptop CPU or, with docker-compose.gpu.yml, on the server's NVIDIA GPU.
ENV BCT_STATIC_DIR=/app/static \
    BCT_ASSETS_DIR=/data/assets \
    BCT_BAKED_ASSETS_DIR=/opt/bct/baked-assets \
    BCT_DOCUMENTS_DIR=/data/documents \
    BCT_DATA_DIR=/data/state \
    BCT_BIND_HOST=0.0.0.0 \
    BCT_BIND_PORT=8000 \
    BCT_INGEST_LOCAL_INDEX=1 \
    BCT_DEFAULT_PROFILE=local_hybrid \
    BCT_PADDLE_PYTHON=/opt/paddle-venv/bin/python \
    HF_HOME=/opt/bct/hf \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1 \
    PYTHONUNBUFFERED=1

# Pre-download embed / rerank / EasyOCR / Docling weights so the first query or upload is not a
# cold start. Docling is warmed by converting a one-line PDF: the exact models ingestion loads.
RUN python -c "from embedding import create_embedding_model; create_embedding_model(); \
from reranker import create_reranker; create_reranker(); \
import easyocr; easyocr.Reader(['ar'], gpu=False, verbose=False); \
import pymupdf; d = pymupdf.open(); d.new_page().insert_text((72, 72), 'warm'); d.save('/tmp/warm.pdf'); \
from ingestion.docling_layout import _convert; _convert('/tmp/warm.pdf'); \
print('local models warmed')"
# PaddleOCR-VL weights too, so a server without internet can read image regions: start the real
# worker once (it loads the pipeline, reports ready) and ask it to quit.
RUN echo '{"cmd": "quit"}' | /opt/paddle-venv/bin/python -u ingestion/paddle_vl_worker.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "docker_serve.py"]
