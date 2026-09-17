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
  <img alt="Neo4j" src="https://img.shields.io/badge/Neo4j-Graph%20Lite-008CC1?style=for-the-badge&logo=neo4j&logoColor=white" />
</p>

---

##  What this is

A research prototype for **Banque Centrale de Tunisie** circulars and notes.

You type a question in French or Arabic. The stack retrieves passages from the PDF corpus, drafts an answer that may only cite retrieved evidence, then opens the real source page in the right panel with the quotation highlighted when the locator finds it.

This package is the consolidated build: FastAPI runtime, Graph Lite, React UI, and incremental PDF ingestion.

> ⚠️ **Not legal advice.** Treat every answer as a research aid. Open the cited page and decide for yourself.

---

##  What you get

| | Capability |
| :--- | :--- |
| <img src="docs/icon-search.svg" width="36" alt="" /> | **Grounded search** across `cloud`, `local_hybrid`, and `local` profiles |
| <img src="docs/icon-evidence.svg" width="36" alt="" /> | **Evidence panel** with the real PDF, physical page, and quote highlight |
| <img src="docs/icon-graph.svg" width="36" alt="" /> | **Graph Lite** Neo4j edges for `CITES`, `AMENDS`, `REPLACES`, `ABROGATES` |
| <img src="docs/icon-ingest.svg" width="36" alt="" /> | **Incremental ingest** for a new PDF without rebuilding the whole corpus |

- 💬 Conversation history in SQLite (reopen from the left sidebar)
- 🔒 Citation checks that block invented filenames and pages
- 📄 Source paths resolved only from trusted document roots / ingestion ledger

---

##  Layout

```text
backend/     FastAPI · RAG · Graph Lite · PDF viewer · ingestion
frontend/    React + TypeScript UI (Vite)
docs/        README icons and crest
```

This package does **not** ship API keys, BCT PDFs, vector assets, Neo4j data, Chroma data, or `node_modules`. You bring those.

---

##  Quick start

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
BCT_DEFAULT_PROFILE=cloud
BCT_BOOTSTRAP_ADMIN_EMAIL=admin@bct.tn
BCT_BOOTSTRAP_ADMIN_PASSWORD=change-me-now
```

| Key | Used for |
| --- | --- |
| `GROQ_API_KEY` | Answer model (+ Graph Lite extraction) |
| `VOYAGE_API_KEY` | Cloud retrieval / rerank |
| `GEMINI_API_KEY` | Visual repair on hard / Arabic pages |
| `BCT_DOCUMENTS_DIR` | Root of original public BCT PDFs |
| `BCT_BOOTSTRAP_ADMIN_EMAIL` / `BCT_BOOTSTRAP_ADMIN_PASSWORD` | Creates the first approved administrator on API startup |

---

### 2️⃣ Graph Lite (optional, recommended) 🕸️

```powershell
$env:BCT_NEO4J_PASSWORD = "choose-a-password"
docker compose -f docker-compose.graph.yml up -d
```

Add to `.env`:

```text
BCT_ENABLE_GRAPH=1
BCT_INGEST_GRAPH=1
BCT_NEO4J_PASSWORD=choose-a-password
```

Bootstrap once from existing runtime chunks:

```powershell
python graph_lite.py bootstrap-assets `
  --native-chunks "C:\path\to\runtime-assets\native.jsonl" `
  --catalog-dir "C:\path\to\your\BCT-PDF-corpus"
```

Trust rules live in [`backend/GRAPH_LITE.md`](backend/GRAPH_LITE.md).

---

### 3️⃣ Start the API 🔌

```powershell
python run_api.py `
  --assets "C:\path\to\runtime-assets" `
  --documents "C:\path\to\your\BCT-PDF-corpus" `
  --enable-graph
```

API default: `http://127.0.0.1:8000`

Authentication uses httpOnly session cookies. New registrations start as `role=user` / `status=pending` until an administrator approves them. PDF upload and runtime configuration live in the administrator dashboard (same login page as normal users).

---

### 4️⃣ Start the UI 🖥️

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
PDF → validate → PyMuPDF extract → StructuredDocument
   → Gemini on Arabic / bad visual pages
   → page-local chunks → Voyage + Chroma + BM25
   → Graph Lite → activate new asset version
```

- HTTP upload: the API reloads retrieval backends after success.
- CLI ingest while the API is running: restart the API so it loads the new asset version.
- A page whose native text contradicts its filename (reversed / font-garbled digits, e.g. `لسنة 6112`) is re-read from the page image by Gemini and the transcription becomes the page's text (`native_replaced_by_gemini`). The garbled native text stays in `structured.json` only.

### Re-extract pages with unreliable digits (staged)

Works on a **candidate** copy of the asset root; the live corpus is untouched until you point the API at the candidate.

```powershell
cd backend
# list affected PDFs/pages in the live corpus
python reingest_unreliable.py --assets "C:\path\to\runtime-assets" --dry-run
# copy the live root, then re-ingest every affected PDF through Gemini
$env:BCT_GEMINI_MODEL = "gemini-3.5-flash-lite"   # free tier: 500 req/day; gemini-3.7-flash is 20/day
python reingest_unreliable.py --assets "C:\path\to\runtime-assets-candidate" --seed-from "C:\path\to\runtime-assets" --documents "C:\path\to\documents"
# compare, then serve from the candidate root
python run_api.py --assets "C:\path\to\runtime-assets-candidate" ...
```

Set `BCT_GEMINI_CACHE` / `BCT_VOYAGE_RUNTIME_CACHE` / `BCT_GOOGLE_RUNTIME_CACHE` to the live caches to reuse transcriptions and embeddings. Cloud indexes are provider-specific (`voyage` vs `google`); do not mix. Rebuild local Chroma separately for the `local` profiles.

---

##  Safety notes

| Rule | Behavior |
| --- | --- |
| 🚫 Path escape | Browser cannot request arbitrary local files |
| 📎 Citations | Built from retrieval metadata, not free-form model filenames |
| 🧪 Staging | New asset versions stage first; a failed activation keeps the old corpus |
| ✅ Graph edges | Stored only after deterministic quote + instrument checks |

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
- Graph Lite policy: [`backend/GRAPH_LITE.md`](backend/GRAPH_LITE.md)
- CI: GitHub Actions runs backend `pytest` and frontend lint/build on `main` and pull requests (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml))

Built as an internship / research prototype. Use it to find and inspect sources, not to replace expert review.
