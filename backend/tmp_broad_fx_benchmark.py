"""Batch-run broad FX/bank questions against the live grounded chat path."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=False)
os.environ.setdefault("BCT_ENABLE_GRAPH", "1")
os.environ.setdefault("BCT_NEO4J_URI", "bolt://127.0.0.1:7687")
os.environ.setdefault("BCT_NEO4J_USERNAME", "neo4j")
os.environ.setdefault("BCT_NEO4J_PASSWORD", "bct-graph-lite-verify")
os.environ.setdefault("BCT_DEFAULT_PROFILE", "cloud")

from ingestion.index import configure_runtime_assets
from conversation import chat
from regulatory_graph_lite.runtime import open_relationship_graph_runtime
from runtime_profiles import RuntimeProfileManager
from runtime_retrieval import create_voyage_backend_from_environment

QUESTIONS = [
    "Quelles opérations les intermédiaires agréés sont-ils autorisés à effectuer ?",
    "Quelles sont les obligations des banques concernant les opérations de change ?",
    "Quelles règles encadrent les garanties bancaires ?",
    "Quelles sont les règles concernant les taux d’intérêt ?",
    "Comment les banques doivent-elles déclarer certaines opérations à la BCT ?",
    "Quelles sont les conditions applicables aux opérations entre intermédiaires agréés ?",
    "Quelles sont les règles concernant les opérations de couverture du risque de change ?",
    "Quelles règles s’appliquent aux opérations de couverture du risque de taux d’intérêt ?",
    "Qu’est-ce que la réglementation prévoit concernant les FRA ?",
    "Quelles sont les conditions applicables aux opérations de change à terme ?",
    "Comment sont encadrées les opérations sur les marchés des changes ?",
    "Quelles règles concernent la détermination ou l'application des taux de change ?",
    "Quelles sont les règles pour un Tunisien qui souhaite ouvrir un compte à l’étranger ?",
    "Dans quels cas une personne physique peut-elle détenir un compte en devises ?",
    "Quelles sont les conditions pour transférer de l’argent à un membre de sa famille à l’étranger ?",
    "Que prévoit la réglementation pour les Tunisiens résidant à l’étranger ?",
    "Quelles sont les règles pour le paiement d’un fournisseur étranger ?",
    "Comment une entreprise tunisienne peut-elle régler une facture en devises ?",
    "Quelles sont les conditions pour importer des marchandises ?",
    "Quelles règles s’appliquent aux exportations et au rapatriement des recettes ?",
    "Une entreprise peut-elle conserver des recettes en devises à l’étranger ?",
    "Quelles sont les règles concernant les avances sur importation ?",
    "Quels documents sont nécessaires pour effectuer un règlement à l’étranger ?",
    "Quelles obligations les banques doivent-elles respecter en matière de change ?",
    "Quelles sont les règles concernant les transferts de fonds vers l’étranger ?",
    "Dans quels cas une autorisation de la Banque Centrale est-elle nécessaire ?",
    "Quelles sont les principales obligations imposées aux intermédiaires agréés ?",
    "Quelles opérations sont soumises à déclaration auprès de la Banque Centrale ?",
    "Quelles sont les règles applicables aux comptes en devises ?",
    "Est-ce qu’une entreprise peut payer un fournisseur à l’étranger ?",
    "Comment ça marche pour envoyer de l’argent à l’étranger ?",
    "Est-ce que les banques peuvent faire cette opération ?",
    "Il faut une autorisation pour ça ?",
    "C’est quoi la limite pour les devises ?",
    "Est-ce qu’un étudiant peut transférer de l’argent pour ses études ?",
    "Qu’est-ce que la BCT dit sur les comptes en devises ?",
    "Est-ce qu’il y a une réglementation pour les opérations de change ?",
    "Quelle est la règle pour les virements vers l’étranger ?",
    "La BCT autorise-t-elle cette opération ?",
]

ASSETS = Path(
    r"C:\Users\Moemen Super\BCT-Regulatory-Search-local-data\session-20260905\runtime-assets"
)
OUT_DIR = Path(
    r"C:\Users\Moemen Super\BCT-Regulatory-Search-local-data\session-20260905\benchmarks"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "broad_fx_graph_benchmark.jsonl"
SUMMARY_PATH = OUT_DIR / "broad_fx_graph_benchmark_summary.json"


def _compact_sources(sources):
    rows = []
    for source in sources or []:
        if hasattr(source, "model_dump"):
            item = source.model_dump()
        elif isinstance(source, dict):
            item = source
        else:
            item = {
                "filename": getattr(source, "filename", None),
                "page": getattr(source, "page", None),
                "quote": getattr(source, "quote", None),
                "excerpt": getattr(source, "excerpt", None),
            }
        rows.append(
            {
                "filename": item.get("filename") or item.get("source"),
                "page": item.get("page"),
                "quote": (item.get("quote") or "")[:240],
                "excerpt": (item.get("excerpt") or "")[:160],
            }
        )
    return rows


def main() -> None:
    configure_runtime_assets(ASSETS, validate=True)
    manager = RuntimeProfileManager(
        None,
        create_voyage_backend_from_environment,
        local_retrieval_factory=None,
    )
    runtime = manager.get("cloud")
    graph_runtime = open_relationship_graph_runtime()
    if graph_runtime is None:
        raise SystemExit("Graph Lite failed to open; aborting benchmark.")
    graph_retriever = graph_runtime.retriever
    print(
        f"graph_ready=True keys={sum(1 for n in ['GROQ_API_KEY',*[f'GROQ_API_KEY_{i}' for i in range(2,8)]] if os.environ.get(n))}",
        flush=True,
    )

    # Resume support: skip questions already written.
    done = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["question"])
            except Exception:
                continue

    counts: dict[str, int] = {}
    with OUT_PATH.open("a", encoding="utf-8") as handle:
        for index, question in enumerate(QUESTIONS, start=1):
            if question in done:
                print(f"[{index}/{len(QUESTIONS)}] skip (done)", flush=True)
                continue
            print(f"[{index}/{len(QUESTIONS)}] {question[:80]}", flush=True)
            started = time.time()
            row = {
                "index": index,
                "question": question,
                "elapsed_s": None,
                "status": None,
                "answer": None,
                "sources": [],
                "graph_trace": {},
                "refusal_reason": None,
                "refusal_diagnostics": [],
                "error": None,
            }
            try:
                result = chat(
                    question,
                    {"turns": []},
                    retrieval_backend=runtime.retrieval_backend,
                    graph_retriever=graph_retriever,
                    llm_provider=runtime.answer_provider,
                )
                row["elapsed_s"] = round(time.time() - started, 2)
                row["status"] = result.get("status")
                row["answer"] = result.get("answer")
                row["sources"] = _compact_sources(result.get("sources"))
                row["graph_trace"] = result.get("graph_trace") or {}
                row["refusal_reason"] = result.get("refusal_reason")
                row["refusal_diagnostics"] = result.get("refusal_diagnostics") or []
            except Exception as error:
                row["elapsed_s"] = round(time.time() - started, 2)
                row["error"] = f"{type(error).__name__}: {error}"
                row["traceback"] = traceback.format_exc()
            counts[row["status"] or "error"] = counts.get(row["status"] or "error", 0) + 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            preview = (row.get("answer") or row.get("error") or "")[:160].replace("\n", " ")
            print(
                f"  -> status={row['status']} sources={len(row['sources'])} "
                f"graph={((row.get('graph_trace') or {}).get('status'))} "
                f"{row['elapsed_s']}s | {preview}",
                flush=True,
            )
            # Ease Voyage rate limits between live questions.
            time.sleep(2.5)

    # Rebuild summary from full file
    status_counts: dict[str, int] = {}
    rows = []
    for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows.append(item)
        key = item.get("status") or ("error" if item.get("error") else "unknown")
        status_counts[key] = status_counts.get(key, 0) + 1
    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "n": len(rows),
                "status_counts": status_counts,
                "out_path": str(OUT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(status_counts, ensure_ascii=False, indent=2), flush=True)
    graph_runtime.close()


if __name__ == "__main__":
    main()
