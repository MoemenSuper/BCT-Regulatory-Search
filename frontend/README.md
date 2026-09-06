# Espace Recherche Réglementaire — Frontend

React + TypeScript + Vite UI for the Banque Centrale de Tunisie regulatory research desk. It implements the three-panel research layout from the design mockups and can call the existing FastAPI `/chat` backend for live answers.

## What this UI is

A professional institutional prototype (not a Gradio shell) with:

| Panel | Role |
| --- | --- |
| **Header** | BCT branding, app title, Signets / Historique, FR \| عربي, researcher menu |
| **Left (~20%)** | Research history (`Nouvelle recherche`, date groups, selected navy card) |
| **Center** | Search bar, regulatory research note, follow-up question box |
| **Right (~29%)** | Evidence / Document tabs, mock PDF viewer, selected passage |

Stack: **React 19**, **TypeScript**, **Vite**, **plain CSS**, **lucide-react** (icons only). No Tailwind, Next.js, Redux, shadcn, or MUI.

## Layout & visual system

- Full-viewport shell: navy header (`#002060`) + light blue-gray workspace
- Thin blue-gray borders, light panel shadows, restrained motion (press feedback, gated hovers, `prefers-reduced-motion`)
- Research note titles use a **serif** face; UI chrome stays sans-serif
- Official BCT logo assets in `public/` (color + white header variant with crest linework preserved)

## Source map

```
frontend/src/
  App.tsx                 # Shell state, search/follow-up → API
  styles.css              # Global institutional styling
  api/chat.ts             # POST /api/chat client
  components/
    Header.tsx
    HistorySidebar.tsx
    SearchBar.tsx
    ResearchNote.tsx
    FollowUpBox.tsx
    EvidencePanel.tsx
    PdfMockViewer.tsx
  data/mockData.ts        # Seed content + helpers to map API → note/evidence
  types/ui.ts
```

## Local interactivity

- History card selection styling
- Preuve / Document tab switching
- PDF zoom percentage (+/−)
- Search (Enter) and **Poser** submit live questions when the backend is up

## Backend wiring

The Gradio ChatInterface is not used by this UI. The React app talks to FastAPI:

1. Vite dev server proxies `/api/*` → `http://127.0.0.1:8000` (see `vite.config.ts`)
2. `POST /api/chat` with `{ "question": "..." }`
3. Response `{ answer, sources[] }` is mapped into:
   - note **Synthèse** / **Sources principales**
   - evidence filename / page from the first source
4. `app.py` allows CORS from the Vite origin for local use

Still mock / later work:

- Real PDF rendering and in-document highlight sync
- Persistent history / bookmarks
- Multi-turn memory in the UI (API currently uses an empty memory state per request)

## How to run

From the **repository root** (not `frontend/`):

```bash
uvicorn app:app --reload --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`).

Useful scripts:

```bash
npm run build    # tsc + production bundle
npm run preview  # serve the production build
```

## Assets

| File | Use |
| --- | --- |
| `public/bct-logo-official.png` | Official color logo |
| `public/bct-logo-white.png` | Header logo (white wordmark, dark crest detail, transparent) |
| `public/favicon.svg` | Browser tab icon |

## Design reference

The visual target is `design-concepts/bct-three-panel-research-desk.png` (and related BCT design crops): dark navy header, three columns, research note with double-ruled metadata, Acrobat-style PDF mark, paper-plane send icon.
