"""Live Suite B retrieval eval against the session runtime-assets index."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ASSETS = Path(
    r"C:\Users\Moemen Super\BCT-Regulatory-Search-local-data"
    r"\session-20260905\runtime-assets"
)
SUITE_B = (
    Path(__file__).resolve().parent / "human_questions.jsonl"
)
OUT = ROOT.parent / "tmp" / "suite_b_live_retrieval_after.json"


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and (key not in os.environ or not os.environ.get(key)):
            os.environ[key] = value


def _top_sources(ranked, limit=5) -> list[str]:
    names = []
    seen = set()
    for doc, _score in ranked[:limit]:
        name = Path(str(doc.metadata.get("source", ""))).name
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _hit(expected: list[str], got: list[str]) -> bool:
    if not expected:
        return True  # clarify / open — retrieval not scored as miss
    want = {Path(x).name for x in expected}
    return bool(want & set(got))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    _load_dotenv(ROOT / ".env")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    from ingestion.index import configure_runtime_assets, resolve_active_assets
    from runtime_retrieval import create_voyage_backend_from_environment

    active = configure_runtime_assets(ASSETS, validate=True)
    print(f"assets={ASSETS}")
    print(f"active={active}")
    print(f"native_bytes={(active / 'native.jsonl').stat().st_size}")

    backend = create_voyage_backend_from_environment()
    print(f"backend={type(backend).__name__}")

    items = [
        json.loads(line)
        for line in SUITE_B.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows = []
    t0 = time.time()
    for index, item in enumerate(items, start=1):
        query = item["turns"][0]
        started = time.time()
        try:
            ranked = list(backend.retrieve(query))
            sources = _top_sources(ranked, limit=5)
            error = ""
        except Exception as exc:  # noqa: BLE001 — eval harness
            ranked = []
            sources = []
            error = f"{type(exc).__name__}: {exc}"
        expected = item["expected"].get("instruments") or []
        scored = bool(expected)
        ok = (not scored) or _hit(expected, sources)
        row = {
            "id": item["id"],
            "category": item["category"],
            "source_hint": item["source_hint"],
            "lang": item.get("lang", "fr"),
            "query": query,
            "expected_instruments": expected,
            "top_sources": sources,
            "scored": scored,
            "pass": ok,
            "error": error,
            "elapsed_s": round(time.time() - started, 2),
            "answer_shape": item["expected"].get("answer_shape"),
        }
        rows.append(row)
        mark = "PASS" if ok else "FAIL"
        if error:
            mark = "ERR"
        print(
            f"[{index}/{len(items)}] {mark} {item['id']} "
            f"hint={item['source_hint']} cat={item['category']} "
            f"got={sources[:3]}",
            flush=True,
        )

    scored_rows = [r for r in rows if r["scored"] and not r["error"]]
    passed = sum(1 for r in scored_rows if r["pass"])
    failed = [r for r in scored_rows if not r["pass"]]
    errors = [r for r in rows if r["error"]]
    unscored = [r for r in rows if not r["scored"]]

    by_cat = defaultdict(lambda: {"n": 0, "pass": 0})
    by_hint = defaultdict(lambda: {"n": 0, "pass": 0})
    for r in scored_rows:
        by_cat[r["category"]]["n"] += 1
        by_hint[r["source_hint"]]["n"] += 1
        if r["pass"]:
            by_cat[r["category"]]["pass"] += 1
            by_hint[r["source_hint"]]["pass"] += 1

    summary = {
        "assets": str(ASSETS),
        "active": str(active),
        "total": len(rows),
        "scored": len(scored_rows),
        "pass": passed,
        "fail": len(failed),
        "errors": len(errors),
        "unscored_clarify_open": len(unscored),
        "pass_rate": round(passed / len(scored_rows), 3) if scored_rows else None,
        "elapsed_s": round(time.time() - t0, 1),
        "by_category": {
            k: {**v, "rate": round(v["pass"] / v["n"], 3) if v["n"] else None}
            for k, v in sorted(by_cat.items())
        },
        "by_source_hint": {
            k: {**v, "rate": round(v["pass"] / v["n"], 3) if v["n"] else None}
            for k, v in sorted(by_hint.items())
        },
        "failures": [
            {
                "id": r["id"],
                "category": r["category"],
                "source_hint": r["source_hint"],
                "query": r["query"],
                "expected": r["expected_instruments"],
                "got": r["top_sources"],
            }
            for r in failed
        ],
        "error_rows": [
            {"id": r["id"], "error": r["error"], "query": r["query"]} for r in errors
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT}")
    return 0 if not failed and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
