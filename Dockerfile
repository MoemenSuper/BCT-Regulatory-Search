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
# Large wheels over slow or filtered networks: wait longer and retry instead of failing the build.
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10
# DEVICE=cpu (default) or gpu (docker-compose.gpu.yml) picks the torch and Paddle builds.
ARG DEVICE=cpu
# Full local_hybrid stack: retrieval (e5/Chroma/BGE), EasyOCR and Docling. Torch goes in first:
# the CPU image takes the CPU-only wheel (~200 MB); the GPU image PyPI's CUDA wheel (several GB).
# Docling pulls the GUI OpenCV build; EasyOCR uses the headless one and both provide cv2, so the
# GUI build is removed and headless reinstalled.
RUN if [ "$DEVICE" = "gpu" ]; then \
        pip install --no-cache-dir torch torchvision ; \
    else \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu ; \
    fi \
    && pip install --no-cache-dir -r requirements-local.txt \
    && pip install --no-cache-dir "filelock>=3.18,<4" "docling>=2.130,<3" \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps \
        "opencv-python-headless==$(python -c 'import importlib.metadata as m; print(m.version("opencv-python-headless"))')"

# PaddleOCR-VL (image regions, scanned French pages) in its own venv: its worker process is the only
# Paddle user, and keeping Paddle out of the torch environment rules out a clash of their GPU
# libraries. DEVICE=gpu builds its CUDA wheel. Same Paddle version on both: PaddleOCR-VL needs >= 3.2.
RUN python -m venv /opt/paddle-venv \
    && if [ "$DEVICE" = "gpu" ]; then \
        /opt/paddle-venv/bin/pip install --no-cache-dir "paddlepaddle-gpu==3.2.1" \
            -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ ; \
    else \
        /opt/paddle-venv/bin/pip install --no-cache-dir "paddlepaddle==3.2.1" \
            -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ ; \
    fi \
    && /opt/paddle-venv/bin/pip install --no-cache-dir "paddleocr[doc-parser]>=3.4,<4"
# PaddleOCR-VL weights (~2 GB), so a server without internet can read image regions. Before the app
# code is copied, so code changes do not download them again. Download only: loading the 0.9B model
# needs more memory than a build may have. Names = pipeline v1.6 in ingestion/paddle_vl_worker.py
# (orientation and unwarping models are off there).
RUN PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1 /opt/paddle-venv/bin/python -c \
    "from paddlex.inference.utils.official_models import official_models; \
[print(official_models[name]) for name in ('PP-DocLayoutV3', 'PaddleOCR-VL-1.6-0.9B')]"

COPY backend/ ./
COPY --from=ui /ui/dist /app/static
# Bake the local documents/ tree into the image (~100MB). Requires documents/ at build time
# (gitignored — present on the builder machine). Recipients get PDFs from the image, no host mount.
COPY documents/ /data/documents/
# Slim local runtime assets (chunks + Chroma). Seeded into the assets volume on first boot.
# Refresh with: python backend/bake_assets.py --source <live asset root>
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
RUN python -c "from rag.embedding import create_embedding_model; create_embedding_model(); \
from rag.reranker import create_reranker; create_reranker(); \
import easyocr; easyocr.Reader(['ar'], gpu=False, verbose=False); \
import pymupdf; d = pymupdf.open(); d.new_page().insert_text((72, 72), 'warm'); d.save('/tmp/warm.pdf'); \
from ingestion.docling_layout import _convert; _convert('/tmp/warm.pdf'); \
print('local models warmed')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "docker_serve.py"]
