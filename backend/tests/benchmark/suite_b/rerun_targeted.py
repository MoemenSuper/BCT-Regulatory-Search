"""Re-run a fixed list of Suite B ids; merge into suite_b_live_answers.json."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ASSETS = Path(
    r"C:\Users\Moemen Super\BCT-Regulatory-Search-local-data"
    r"\session-20260905\runtime-assets"
)
SUITE_B = Path(__file__).resolve().parent / "human_questions.jsonl"
OUT = ROOT.parent / "tmp" / "suite_b_live_answers.json"

# Wrong-year cites + prior 429s + near-miss content that may flip with newest pack
IDS = [
    "B-020",
    "B-124",
    "B-042",
    "B-081",
    "B-085",
    "B-094",
    "B-116",
    "B-131",
]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_live_answers import (  # noqa: E402
    _load_dotenv,
    _score_item,
    _summarize,
    _top_sources,
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    _load_dotenv(ROOT / ".env")
    os.chdir(ROOT)
    os.environ.setdefault("BCT_VOYAGE_RETRY_SWEEPS", "3")
    pause = float(os.environ.get("BCT_EVAL_PAUSE_SECONDS", "4"))

    from conversation import chat
    from ingestion.index import configure_runtime_assets
    from jsonl_supersession import clear_supersession_cache
    from runtime_retrieval import create_voyage_backend_from_environment

    data = json.loads(OUT.read_text(encoding="utf-8"))
    items = {
        json.loads(line)["id"]: json.loads(line)
        for line in SUITE_B.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    print(f"rerunning {len(IDS)} targeted fails", flush=True)

    clear_supersession_cache()
    configure_runtime_assets(ASSETS, validate=True)
    backend = create_voyage_backend_from_environment()
    by_id = {row["id"]: row for row in data["rows"]}
    order = [row["id"] for row in data["rows"]]

    for index, qid in enumerate(IDS, start=1):
        item = items[qid]
        query = item["turns"][0]
        started = time.time()
        error = ""
        try:
            ranked = list(backend.retrieve(query))
            top = _top_sources(ranked, limit=5)
            result = chat(query, memory_state={}, retrieval_backend=backend)
            status = result.get("status") or "answered"
            answer = result.get("answer") or ""
            sources = result.get("sources") or []
        except Exception as exc:  # noqa: BLE001
            ranked = []
            top = []
            status = "error"
            answer = ""
            sources = []
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(20)

        score = _score_item(
            item, status=status, answer=answer, sources=sources, top=top
        )
        row = {
            "id": qid,
            "category": item.get("category"),
            "lang": item.get("lang"),
            "query": query,
            "expected": item.get("expected"),
            "status": status,
            "answer": answer[:500],
            "top_sources": top,
            "answer_sources": [
                Path(str(s.get("file") or s.get("source") or "")).name
                for s in sources
            ],
            "error": error,
            "elapsed_s": round(time.time() - started, 2),
            **score,
        }
        by_id[qid] = row
        mark = (
            "PASS"
            if score["retrieval_ok"] and score["answer_ok"] and not error
            else ("ERR" if error else "FAIL")
        )
        print(
            f"[{index}/{len(IDS)}] {mark} {qid} status={status} "
            f"ret={int(score['retrieval_ok'])} ans={int(score['answer_ok'])} "
            f"inst={int(score['instrument_in_sources'])} "
            f"{row['elapsed_s']}s missing={score['missing_touch']} "
            f"src={row['answer_sources'][:2]}",
            flush=True,
        )
        time.sleep(pause)

    rows = sorted(
        by_id.values(),
        key=lambda row: order.index(row["id"]) if row["id"] in order else 999,
    )
    summary = _summarize(rows)
    summary["complete"] = True
    summary["rerun_targeted"] = IDS
    payload = {"summary": summary, "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
