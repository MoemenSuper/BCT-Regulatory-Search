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
backend/cloud/  optional cloud profile only (Voyage search, Gemini page reading)
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
- Bakes a slim local Chroma runtime index (`baked-runtime-assets/`) and seeds it into the assets volume on first boot so search works without re-uploading the 454 PDFs
- Serves the app at **http://localhost:8080**

### What you must do

1. Install **Docker Desktop** (or Docker Engine + Compose).
2. Get the project folder **with `documents/` and `baked-runtime-assets/` in it** (both are gitignored: they are handed over with the project folder, not stored in git). Keep this folder on the server: it is the code, and the image is built from it. To ship a newer corpus, bake it from a running asset root (only the live version and its search index; a previous bake is renamed, not deleted):

```powershell
cd backend
python bake_assets.py --source "C:\path\to\runtime-assets"
```

3. Copy the env template and fill **required** values:

```powershell
copy .env.example .env
```

Edit `.env` and set at least:

| Variable | Meaning |
| --- | --- |
| `BCT_BOOTSTRAP_ADMIN_EMAIL` | First admin login email |
| `BCT_BOOTSTRAP_ADMIN_PASSWORD` | Strong password for that admin |
| `OPENAI_API_KEY` **or** `ANTHROPIC_API_KEY` **or** `GEMINI_API_KEY` **or** `GROQ_API_KEY` | The answer model. **One key is enough**: the app answers with the first provider that has a key (OpenAI, Claude, Groq, then Gemini), or the one set in `BCT_ANSWER_PROVIDER`. Provider, keys and models can all be changed later in **Admin → Configuration**. |
| `VOYAGE_API_KEY` | Cloud-profile search / rerank only (optional for local_hybrid) |

Default models: OpenAI `gpt-6.1-sol`, Claude `claude-sonnet-5-5`, Gemini `gemini-3.8-flash`, Groq `openai/gpt-oss-120b` (override with `BCT_OPENAI_MODEL`, `BCT_ANTHROPIC_MODEL`, `BCT_GEMINI_ANSWER_MODEL`, `BCT_GROQ_MODEL`). Every model call (routing the question, picking the evidence, writing the answer, naming the conversation) uses the chosen provider.

Recipients who only pull/run a pre-built image do **not** need a separate PDF folder or a multi-hour ingest — documents and local indexes ship in the image.

4. Start everything. Two cases:

   - **A. The server already runs Nginx** (most likely): start the app, then add the Nginx file as in [Behind Nginx](#behind-nginx-red-hat-server).

     ```bash
     docker compose up -d --build
     ```

   - **B. Nothing is installed but Docker**: the same command with `--profile nginx` also starts an Nginx container in front of the app. See [Server with nothing installed](#server-with-nothing-installed-nginx-included).

     ```bash
     docker compose --profile nginx up -d --build
     ```

5. Open **http://localhost:8080** on the server itself (or the server's address in case B), sign in with the bootstrap admin account.
6. Optional: upload more PDFs in the admin UI. The 454 PDFs baked into the image are already searchable (the Overview counts them); upload only new ones.
7. Approve other user accounts when they register.

**Rebuilt image, old volume.** The index lives in the `bct-assets` volume, which outlives the image (Docker names volumes after the project folder, so a fresh clone in a folder of the same name reuses them). On start, an image that ships a different index replaces the volume's one (the old one is kept aside in `replaced-…`), and PDFs uploaded earlier are queued again and indexed on top in the background. Nothing to delete by hand. To override PDFs with a host folder, set `BCT_DOCUMENTS_HOST` and uncomment the bind mount in `docker-compose.yml`.

**Something wrong?** Every question (status, reason, seconds), upload and failure is logged, never key values. Send this file to the developer:

```powershell
docker compose logs app > bct-log.txt
```

The same lines are kept in `/data/state/logs/bct.log` in the app-data volume, so they survive a restart and a re-created container (`docker compose cp app:/data/state/logs/bct.log .` copies it out). What to look for first:

- `Startup check:` — written at every start. It names what is missing: no answer-model key, an empty index, no admin account, too little memory or disk.
- `Unexpected error <reference>` — any crash, with its full error. The user sees the same reference on screen (*Unexpected server error (reference 3f9a2c1b)*): search the log for it.
- `WARNING` and `ERROR` lines — a failed upload or page reading, an answer model that refused the key or was out of quota, an index that could not be updated.

If the log says *the GPU driver was reset* (Windows laptops under heavy load), restart the app: a process cannot use the GPU again after a driver reset.

### NVIDIA GPU server

With the NVIDIA driver and NVIDIA Container Toolkit installed on the host:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

This builds torch and PaddleOCR-VL with CUDA and gives the container the GPUs; search, reranking, EasyOCR and Docling pick the GPU up by themselves. Without a GPU, the default command above runs everything on CPU (same answers; reading scanned or picture-heavy uploads is slower).

### Relations between texts (abrogations, replacements)

When a PDF says it abrogates, replaces or modifies another text, the app records that relation and search uses it at once (an answer about the old text also brings up the new one). **Admin → Relations between texts** lists every relation as *To review*, with the page that states it opened and the sentence highlighted. The admin approves or rejects each one, and can add a missing one (new text, page, type, old text): the app checks that the page names the old text and keeps that sentence as the proof. Decisions are stored apart from the index, so re-uploading a PDF or shipping a new index keeps them; every decision is in the audit log. A relation proves "X replaced Y", never "X is in force".

### Speed on a server without GPU

On CPU almost all of an answer's time is the reranker (about 0.4–0.9 s per passage; the answer model itself takes a few seconds). Two things keep it down:

- Each candidate passage is reranked once, against the question's wording in its own script (French/Latin or Arabic), instead of against every wording. Same retrieval benchmark result (82 % top-5), 2.3× fewer passages to rerank. Always on.
- **Admin → Configuration → `BCT_SPEED_MODE`**: *Automatic* (default, best quality) or *Fast* (the reranker in 8 bits on CPU: about 2× faster again, but 77 % instead of 82 % on the benchmark). No effect on a GPU server.

Measured on a laptop CPU (no GPU), same questions: a typical question took 104–141 s before, 25–31 s in Automatic and 13–19 s in Fast; a question naming a circular (which adds that circular's passages) 526 s before, 94 s and 40 s.

### Memory

Search and answers run in about 3 GB. Indexing an uploaded PDF starts the Docling layout reader, which needs about **4 GB free** on top: give Docker at least **8 GB**. With less, uploads wait in the queue (the admin Documents page says how much memory is free and how much is needed) instead of crashing the server (`BCT_DOCLING_MIN_FREE_GB` changes the threshold). Reading pictures and scans with PaddleOCR-VL on **CPU** needs about 8 GB of free memory on top, so give Docker at least **12–16 GB** (Docker Desktop: Settings → Resources). With less, the app keeps running: the reader does not start, uploads stay searchable from their PDF text, and the admin page lists those pages as not read visually (`BCT_PADDLE_MIN_FREE_GB` changes the threshold). On a GPU server the model loads into the graphics card instead.

### Servers without internet

Building the image downloads Python packages, the models and the base images, so the server needs internet while it builds (afterwards it runs without). Only if it never has internet: build on a connected machine, `docker save bct-regulatory-search:pilot nginx:1.30-alpine -o bct.tar`, copy `bct.tar` with the project folder, `docker load -i bct.tar` on the server, and start without `--build`.

### Behind Nginx (Red Hat server)

The container is published on `127.0.0.1:8080` only, so on a server the way in is the server's Nginx, which adds HTTPS and one address for everyone (`https://recherche.bct.tn` in the example).

1. Start the app as above (`docker compose up -d`, or `--no-build` with a loaded image).
2. Copy [`deploy/nginx/bct-regulatory-search.conf`](deploy/nginx/bct-regulatory-search.conf) to `/etc/nginx/conf.d/`, and set `server_name` and the two certificate paths to the real ones.
3. Red Hat runs SELinux, which forbids Nginx to open connections by default (users would get *502 Bad Gateway*). Allow it once, then check and reload Nginx:
   ```bash
   sudo setsebool -P httpd_can_network_connect 1
   ```
   ```bash
   sudo nginx -t && sudo systemctl reload nginx
   ```
4. Open HTTPS in the firewall if it is not already: `sudo firewall-cmd --permanent --add-service=https && sudo firewall-cmd --reload`.
5. In `.env`, set `BCT_COOKIE_SECURE=1` (session cookies only travel over HTTPS), then `docker compose up -d`.

What the Nginx file sets for this app, and why (each one breaks something if left at Nginx's default):

| Setting | Value | Without it |
| --- | --- | --- |
| `proxy_read_timeout` | 300 s | answers longer than 60 s (CPU servers: 1–2 min) fail with *504 Gateway Timeout* |
| `client_max_body_size` | 50 MB | PDF uploads over 1 MB fail with *413 Request Entity Too Large* |
| `X-Forwarded-For` / `X-Real-IP` | the user's address | every user shares Nginx's address, so 5 wrong passwords from anyone lock everyone out for 15 minutes |
| `proxy_request_buffering off` | | large uploads are first written to Nginx's own disk |

The compose file sets `FORWARDED_ALLOW_IPS: "*"` so the app trusts the address Nginx forwards; that is safe because only the server itself can reach port 8080.

### Server with nothing installed (Nginx included)

For a Red Hat server with no Nginx of its own: Docker runs both the app and an Nginx in front of it. Nothing else is installed on the server.

1. Install Docker Engine with its Compose plugin (Red Hat's own `podman` does not run this compose file), then start it:
   ```bash
   sudo dnf -y install dnf-plugins-core
   ```
   ```bash
   sudo dnf config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo
   ```
   ```bash
   sudo dnf -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin
   ```
   ```bash
   sudo systemctl enable --now docker
   ```
2. Copy the project folder to the server (with `documents/` and `baked-runtime-assets/`) and fill `.env` as in [What you must do](#what-you-must-do).
3. HTTPS: put the server's certificate in `deploy/nginx/certs/` as `bct.crt` (with its chain) and `bct.key`, and set `BCT_COOKIE_SECURE=1` in `.env`. Without a certificate the Nginx serves plain HTTP (its log says so): keep `BCT_COOKIE_SECURE=0`, or nobody can sign in. Plain HTTP is for a first test only, since passwords then cross the network unencrypted.
4. Open the web ports in the firewall:
   ```bash
   sudo firewall-cmd --permanent --add-service=http --add-service=https && sudo firewall-cmd --reload
   ```
5. Start, from the project folder:
   ```bash
   sudo docker compose --profile nginx up -d --build
   ```
6. Open `https://<server address>` (or `http://` without a certificate).

The Nginx container uses the same settings as the file for a server's own Nginx (50 MB uploads, 5-minute answers, the user's real address), from `deploy/nginx/docker/`. If something else on the server already uses port 80 or 443, set `BCT_HTTP_PORT` / `BCT_HTTPS_PORT` in `.env`. `docker compose logs nginx` shows whether it started with HTTPS. To update later: replace the project folder (keep `.env` and `deploy/nginx/certs/`), then run the start command again.

### Optional later
- To open port 8080 to the network without Nginx (a quick test on a LAN), set `BCT_PUBLISH_ADDRESS=0.0.0.0` in `.env`.

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
OPENAI_API_KEY=...        (or ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY: one is enough)
VOYAGE_API_KEY=...        (cloud profile only)
BCT_DOCUMENTS_DIR=C:\path\to\your\BCT-PDF-corpus
BCT_DEFAULT_PROFILE=local_hybrid
BCT_BOOTSTRAP_ADMIN_EMAIL=admin@bct.tn
BCT_BOOTSTRAP_ADMIN_PASSWORD=change-me-now
```

| Key | Used for |
| --- | --- |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `GROQ_API_KEY` | Answer model: the first provider with a key, or `BCT_ANSWER_PROVIDER`; changeable in Admin → Configuration |
| `VOYAGE_API_KEY` | Cloud retrieval / rerank |
| `GEMINI_API_KEY` | Also: cloud-profile reading of hard / Arabic / chart pages |
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

API default: `http://127.0.0.1:8000`. Add `--enable-ingestion` to allow PDF uploads from the admin page. The log is in `<data dir>/logs/bct.log` (`.demo-data` by default).

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
Upload (seconds): validate → keep the PDF → queue it                        status: queued

Indexing (background worker in the API process, up to 25 queued PDFs per batch)
each PDF: Docling layout + PDF text → StructuredDocument (visual pages marked pending) → chunks
   (a PDF that fails is marked failed with its reason; the rest of the batch goes on)
then once per batch: Chroma + BM25 + supersession_edges.jsonl → activate one new asset version
                                                                          status: enriching / ready

Enrichment (background worker in the API process, one page at a time)
pending pages, unreadable first → EasyOCR (Arabic) / PaddleOCR-VL (charts·tables·hard pages) locally,
   Gemini VLM on cloud → per-page checkpoint in the ingestion ledger
   → every 8–20 pages: re-extract with the read pages → staged activation → search reload
   → status: ready, or ready_degraded when pages failed after 3 attempts (admin "Re-read" re-queues them)
```

- HTTP upload returns as soon as the PDF is queued, so hundreds of PDFs upload in minutes; the index is rebuilt once per batch, not once per PDF. The admin Documents tab lists queued, indexing and failed PDFs (with the reason) and polls while anything is queued or `enriching`. A PDF being indexed when the server stops is queued again; one interrupted twice is marked failed. Queued or failed PDFs can be deleted from the list.
- Chat has priority: the worker waits between pages while a question is answered. A page already on the GPU finishes first (PaddleOCR-VL ≈ 100–140 s per chart page on an 8 GB laptop GPU), so a question asked mid-page can be slower.
- Restarts are safe: read pages are checkpointed, the worker resumes with the next pending page, and the PaddleOCR-VL worker process exits with its parent (no orphan holding GPU memory).
- Any hardware: one PaddleOCR-VL worker per API, capped with `FLAGS_gpu_memory_limit_mb` (default total VRAM − 1 GB; `BCT_PADDLE_GPU_MEMORY_MB` overrides, `0` = no cap). The model needs ~7 GB; when the GPU worker cannot start (small GPU, no CUDA) it falls back to CPU — same model, ≈ 11 min per chart page instead of ≈ 1.5–2 min. EasyOCR also falls back to CPU. Models are released when the queue is empty.
- Circuit breaker: 3 consecutive page failures (e.g. the OCR worker crashing on VRAM exhaustion) pause visual reading for 10 minutes and free the models; the PDF stays searchable meanwhile.
- CLI: `python ingest.py …` runs the quick pass and then enriches inline; `--quick-only` leaves the pending pages to a running API started with `--enable-ingestion`. Restart the API after a CLI ingest so it loads the new asset version.
- Tuning: `BCT_INGEST_BATCH` (25 PDFs per index rebuild), `BCT_ENRICH_BATCH_PAGES` (20), `BCT_ENRICH_BATCH_SECONDS` (600), `BCT_ENRICH_IDLE_SECONDS` (3, quiet time after a chat), `BCT_ENRICH_MAX_ATTEMPTS` (3), `BCT_ENRICH_BREAKER_FAILURES` (3), `BCT_ENRICH_COOLDOWN_SECONDS` (600); `BCT_ENRICHMENT=0` disables the worker.
- A page whose native text contradicts its filename (reversed / font-garbled digits, e.g. `لسنة 6112`) is re-read from the page image by the active visual backend and the transcription becomes the page's text (`native_replaced_by_visual`). The garbled native text stays in `structured.json` only.

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
| `cloud` | `BCT_CLOUD_RETRIEVAL_PROVIDER=voyage` (Context-4 + Voyage rerank) or `google` (Gemini embed + Vertex Ranking); indexes are separate | the chosen answer model (OpenAI, Claude, Gemini or Groq) |
| `local_hybrid` | Local E5/BGE | the chosen answer model (OpenAI, Claude, Gemini or Groq) |
| `local` | Local E5/BGE | Ollama (experimental) |

---

## 📎 More

- Frontend details: [`frontend/README.md`](frontend/README.md)
- Domain wording: [`CONTEXT.md`](CONTEXT.md)
- CI: GitHub Actions runs backend `pytest` and frontend lint/build on `main` and pull requests (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml))

Built as an internship / research prototype. Use it to find and inspect sources, not to replace expert review.
