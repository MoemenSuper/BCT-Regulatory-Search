<p align="center">
  <img src="docs/bct-crest.png" alt="Banque Centrale de Tunisie crest" width="296" />
</p>

<h1 align="center"> BCT Regulatory Search </h1>

<p align="center">
  <strong>Ask a BCT regulation question. Get a grounded answer you can open on the cited PDF page.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img alt="React" src="https://img.shields.io/badge/React-UI-61DAFB?style=for-the-badge&logo=react&logoColor=0B1526" />
</p>

---

##  What this is

A research prototype for **Banque Centrale de Tunisie** circulars and notes.

You type a question in French or Arabic. The stack retrieves passages from the PDF corpus, drafts an answer that may only cite retrieved evidence, then opens the real source page in the right panel with the quotation highlighted when the locator finds it.

This package is the consolidated build: FastAPI runtime, React UI, incremental PDF ingestion, and JSONL supersession edges.

> ⚠️ **Not legal advice.** Treat every answer as a research aid. Open the cited page and decide for yourself.

---

##  What you get

| | Capability |
| :--- | :--- |
| <img src="docs/icon-search.svg" width="36" alt="" /> | **Grounded search** across `cloud`, `local_hybrid`, and `local` profiles |
| <img src="docs/icon-evidence.svg" width="36" alt="" /> | **Evidence panel** with the real PDF, physical page, and quote highlight |
| <img src="docs/icon-graph.svg" width="36" alt="" /> | **JSONL supersession** pins successor pages via `supersession_edges.jsonl` |
| <img src="docs/icon-ingest.svg" width="36" alt="" /> | **Incremental ingest** for a new PDF without rebuilding the whole corpus |

- 💬 Conversation history in SQLite (reopen from the left sidebar)
- 🔒 Citation checks that block invented filenames and pages
- 📄 Source paths resolved only from trusted document roots / ingestion ledger

---

##  Layout

```text
backend/     FastAPI · RAG · JSONL supersession · PDF viewer · ingestion
frontend/    React + TypeScript UI (Vite)
docs/        README icons and crest
```

This package does **not** ship API keys, BCT PDFs, vector assets, Chroma data, or `node_modules`. You bring those.

---

##  Docker pilot (recommended for administrators)

One command starts the **UI + API**.

### What Docker does for you
- Installs dependencies for **local_hybrid**: e5 embeddings, BGE reranker, Chroma, EasyOCR, Docling, PaddleOCR-VL (in its own environment)
- Pre-downloads every model (embed, rerank, EasyOCR, Docling, PaddleOCR-VL) into the image: once built, it runs without internet
- Detects the hardware at run time: CPU anywhere, the NVIDIA GPU when started with `docker-compose.gpu.yml`
- Builds the React UI into the API image
- Bakes the local `documents/` PDF corpus into the image (~100 MB)
- Bakes a slim local Chroma runtime index (`baked-runtime-assets/`) and seeds it into the assets volume on first boot so search works without re-uploading the 445 PDFs
- Serves the app at **http://localhost:8080**

### What you must do

1. Install **Docker Desktop** (or Docker Engine + Compose).
2. Get the project (clone or unzip) **with `documents/` and `baked-runtime-assets/` present on the machine that builds the image** (both are gitignored). Export indexes with:

```powershell
cd backend
python tmp\rebuild_local_baked.py
```

(Or re-export from a live assets root: `python tmp\export_baked_assets.py --source "C:\path\to\runtime-assets" --dest "..\baked-runtime-assets"`.)

3. Copy the env template and fill **required** values:

```powershell
copy .env.example .env
```

Edit `.env` and set at least:

| Variable | Meaning |
| --- | --- |
| `BCT_BOOTSTRAP_ADMIN_EMAIL` | First admin login email |
| `BCT_BOOTSTRAP_ADMIN_PASSWORD` | Strong password for that admin |
| `GROQ_API_KEY` | Answer model (required for `local_hybrid` / `cloud`) |
| `VOYAGE_API_KEY` | Cloud-profile search / rerank only (optional for local_hybrid) |
| `GEMINI_API_KEY` | Cloud-profile visual repair only (optional for local_hybrid; Docker uses EasyOCR + Paddle) |

Recipients who only pull/run a pre-built image do **not** need a separate PDF folder or a multi-hour ingest — documents and local indexes ship in the image.

4. Start everything:

```powershell
docker compose up -d --build
```

5. Open **http://localhost:8080**, sign in with the bootstrap admin account.
6. Optional: ingest additional PDFs in the admin UI (incremental). With baked assets, the corpus is already searchable.
7. Approve other user accounts when they register.

If you previously started Compose with an empty `bct-assets` volume and want the baked corpus, remove that volume once (`docker volume rm …`) so first boot can seed again. To override PDFs with a host folder, set `BCT_DOCUMENTS_HOST` and uncomment the bind mount in `docker-compose.yml`.

### NVIDIA GPU server

With the NVIDIA driver and NVIDIA Container Toolkit installed on the host:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

This builds PaddleOCR-VL with its CUDA wheel and gives the container the GPUs; search, reranking, EasyOCR and Docling pick the GPU up by themselves. Without a GPU, the default command above runs everything on CPU (same answers; reading scanned or picture-heavy uploads is slower).

### Servers without internet

Build the image on a connected machine, then move it: `docker save bct-regulatory-search:pilot -o bct.tar`, copy `bct.tar`, `docker load -i bct.tar` on the server, and start it with `docker compose up -d` (no `--build`).

### Optional later
- Behind HTTPS, set `BCT_COOKIE_SECURE=1` in `.env` and restart.

---

##  Quick start (developers, without Docker)

### 1️⃣ Backend

Use Python **3.12** for the full prototype.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-ingestion.txt
```

Create `backend/.env` and fill what you use:

```text
GROQ_API_KEY=...
VOYAGE_API_KEY=...
GEMINI_API_KEY=...
BCT_DOCUMENTS_DIR=C:\path\to\your\BCT-PDF-corpus
BCT_DEFAULT_PROFILE=local_hybrid
BCT_BOOTSTRAP_ADMIN_EMAIL=admin@bct.tn
BCT_BOOTSTRAP_ADMIN_PASSWORD=change-me-now
```

| Key | Used for |
| --- | --- |
| `GROQ_API_KEY` | Answer model |
| `VOYAGE_API_KEY` | Cloud retrieval / rerank |
| `GEMINI_API_KEY` | Cloud-profile visual repair on hard / Arabic / chart pages |
| `BCT_DOCUMENTS_DIR` | Root of original public BCT PDFs |
| `BCT_BOOTSTRAP_ADMIN_EMAIL` / `BCT_BOOTSTRAP_ADMIN_PASSWORD` | Creates the first approved administrator on API startup |

Ingest also merges amendment language into `supersession_edges.jsonl` inside the staged asset version (optional pin at retrieve time).

---

### 2️⃣ Start the API 🔌

```powershell
python run_api.py `
  --assets "C:\path\to\runtime-assets" `
  --documents "C:\path\to\your\BCT-PDF-corpus"
```

API default: `http://127.0.0.1:8000`

Authentication uses httpOnly session cookies. New registrations start as `role=user` / `status=pending` until an administrator approves them. PDF upload and runtime configuration live in the administrator dashboard (same login page as normal users).

---

### 3️⃣ Start the UI 🖥️

```powershell
cd ..\frontend
npm ci
npm run dev
```

Open the Vite URL (usually `http://localhost:5173`). The UI proxies `/api/*` to the backend.

---

## 📥 Add a new PDF

```powershell
cd ..\backend
python ingest.py "C:\path\to\new_circular.pdf" --assets "C:\path\to\runtime-assets"
```

Flow:

```text
Quick pass (seconds, inside the upload request)
PDF → validate → PyMuPDF native text → StructuredDocument (visual pages marked pending)
   → page-local chunks → Voyage + Chroma + BM25
   → merge supersession_edges.jsonl → activate new asset version      status: enriching (searchable)

Enrichment (background worker in the API process, one page at a time)
pending pages, unreadable first → EasyOCR (Arabic) / PaddleOCR-VL (charts·tables·hard pages) locally,
   Gemini VLM on cloud → per-page checkpoint in the ingestion ledger
   → every 8–20 pages: re-extract with the read pages → staged activation → search reload
   → status: ready, or ready_degraded when pages failed after 3 attempts (admin "Re-read" re-queues them)
```

- HTTP upload returns as soon as the quick pass is live; the admin Documents tab shows progress and an ETA, polling while anything is `enriching`.
- Chat has priority: the worker waits between pages while a question is answered. A page already on the GPU finishes first (PaddleOCR-VL ≈ 100–140 s per chart page on an 8 GB laptop GPU), so a question asked mid-page can be slower.
- Restarts are safe: read pages are checkpointed, the worker resumes with the next pending page, and the PaddleOCR-VL worker process exits with its parent (no orphan holding GPU memory).
- Any hardware: one PaddleOCR-VL worker per API, capped with `FLAGS_gpu_memory_limit_mb` (default total VRAM − 1 GB; `BCT_PADDLE_GPU_MEMORY_MB` overrides, `0` = no cap). The model needs ~7 GB; when the GPU worker cannot start (small GPU, no CUDA) it falls back to CPU — same model, ≈ 11 min per chart page instead of ≈ 1.5–2 min. EasyOCR also falls back to CPU. Models are released when the queue is empty.
- Circuit breaker: 3 consecutive page failures (e.g. the OCR worker crashing on VRAM exhaustion) pause visual reading for 10 minutes and free the models; the PDF stays searchable meanwhile.
- CLI: `python ingest.py …` runs the quick pass and then enriches inline; `--quick-only` leaves the pending pages to a running API started with `--enable-ingestion`. Restart the API after a CLI ingest so it loads the new asset version.
- Tuning: `BCT_ENRICH_BATCH_PAGES` (20), `BCT_ENRICH_BATCH_SECONDS` (600), `BCT_ENRICH_IDLE_SECONDS` (3, quiet time after a chat), `BCT_ENRICH_MAX_ATTEMPTS` (3), `BCT_ENRICH_BREAKER_FAILURES` (3), `BCT_ENRICH_COOLDOWN_SECONDS` (600); `BCT_ENRICHMENT=0` disables the worker.
- Tracing (optional): set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL`. Each PDF version is one Langfuse session (`ingest-<sha256>`): the `ingest-document` quick-pass trace, one `enrich-page` trace per page (engine, seconds, chat wait, cold start), `activate-enrichment` per batch, and `open-circuit-breaker` warnings.
- A page whose native text contradicts its filename (reversed / font-garbled digits, e.g. `لسنة 6112`) is re-read from the page image by the active visual backend and the transcription becomes the page's text (`native_replaced_by_visual`). The garbled native text stays in `structured.json` only.

### Re-extract pages with unreliable digits (staged)

Works on a **candidate** copy of the asset root; the live corpus is untouched until you point the API at the candidate.

```powershell
cd backend
# list affected PDFs/pages in the live corpus
python reingest_unreliable.py --assets "C:\path\to\runtime-assets" --dry-run
# copy the live root, then re-ingest every affected PDF through Gemini
$env:BCT_GEMINI_MODEL = "gemini-3.5-flash-lite"   # free tier: 500 req/day; default ingest model is gemini-3.8-flash (falls back to 3.6 on quota)
python reingest_unreliable.py --assets "C:\path\to\runtime-assets-candidate" --seed-from "C:\path\to\runtime-assets" --documents "C:\path\to\documents"
# compare, then serve from the candidate root
python run_api.py --assets "C:\path\to\runtime-assets-candidate" ...
```

Set `BCT_GEMINI_CACHE` / `BCT_VOYAGE_RUNTIME_CACHE` to the live caches to reuse transcriptions and embeddings. Rebuild local Chroma separately for the `local` profiles.

---

##  Safety notes

| Rule | Behavior |
| --- | --- |
| 🚫 Path escape | Browser cannot request arbitrary local files |
| 📎 Citations | Built from retrieval metadata, not free-form model filenames |
| 🧪 Staging | New asset versions stage first; a failed activation keeps the old corpus |
| ✅ Supersession | `supersession_edges.jsonl` pins successor pages; does not prove “in force” |

---

## 🧩 Runtime profiles

Set `BCT_DEFAULT_PROFILE` before startup:

| Profile | Retrieval | Answer |
| --- | --- | --- |
| `cloud` | `BCT_CLOUD_RETRIEVAL_PROVIDER=voyage` (Context-4 + Voyage rerank) or `google` (Gemini embed + Vertex Ranking); indexes are separate | Groq |
| `local_hybrid` | Local E5/BGE | Groq |
| `local` | Local E5/BGE | Ollama (experimental) |

---

## 📎 More

- Frontend details: [`frontend/README.md`](frontend/README.md)
- Domain wording: [`CONTEXT.md`](CONTEXT.md)
- CI: GitHub Actions runs backend `pytest` and frontend lint/build on `main` and pull requests (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml))

Built as an internship / research prototype. Use it to find and inspect sources, not to replace expert review.
