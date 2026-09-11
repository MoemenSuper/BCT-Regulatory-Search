# BCT Regulatory Search frontend

React + TypeScript + Vite interface for the final BCT regulatory-search prototype.

This is no longer a screenshot-only mock. It is wired to the FastAPI backend for:

- new searches and follow-up questions;
- persistent conversation history;
- grounded sources returned by the backend;
- the real cited PDF and physical page;
- highlighted quoted evidence;
- full-document viewing.

## Run

Start the backend first, then:

```powershell
npm ci
npm run dev
```

Vite proxies `/api/*` to `http://127.0.0.1:8000`.

Useful checks:

```powershell
npm run build
npm run lint
```

## Main files

```text
src/App.tsx                     application state + real API flow
src/api/chat.ts                 chat/history/source API client
src/components/HistorySidebar.tsx
src/components/ResearchNote.tsx
src/components/EvidencePanel.tsx
src/data/presentation.ts        maps backend data into the research-note UI
src/styles.css
```

`EvidencePanel` uses backend-rendered source pages for the **Preuve** tab and the original PDF for the **Document** tab. No fake PDF content is used.
