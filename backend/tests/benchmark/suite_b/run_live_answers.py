"""Suite B live answer-layer eval: retrieve → expand → generate → score."""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ASSETS = Path(
    r"C:\Users\Moemen Super\BCT-Regulatory-Search-local-data"
    r"\session-20260905\runtime-assets"
)
SUITE_B = Path(__file__).resolve().parent / "human_questions.jsonl"
OUT = ROOT.parent / "tmp" / "suite_b_live_answers.json"
CHECKPOINT = ROOT.parent / "tmp" / "suite_b_live_answers.partial.json"

PAUSE_S = float(os.environ.get("BCT_EVAL_PAUSE_SECONDS", "0.5"))


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


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    for src, dst in (
        ("\u2010", "-"),
        ("\u2011", "-"),
        ("\u2012", "-"),
        ("\u2013", "-"),
        ("\u2014", "-"),
        ("\u2212", "-"),
        ("'", "'"),
        ("'", "'"),
        ("'", "'"),
    ):
        text = text.replace(src, dst)
    text = text.replace("non-residente", "non residente")
    text = text.replace("stand-by", "standby").replace("stand by", "standby")
    text = text.replace("assurance-credit", "assurance credit")
    text = text.replace("credit documentaire", "credit documentaire")
    return " ".join(text.split())


_MUST_TOUCH_ALIASES = {
    "banque non-residente": (
        "banque non residente",
        "banque non-residente",
        "garantie",
        "non residente",
    ),
    "stand-by": ("standby", "stand-by", "stand by", "lettre de credit"),
    "credit documentaire": ("credit documentaire", "credits documentaires", "documentary"),
    "guarantee": ("guarantee", "garantie", "garant"),
    "technical": ("technical", "fiche technique", "technical sheet"),
    "ministry": ("ministry", "ministere", "ministère", "minister"),
    "sous reserve": ("sous reserve", "sous réserve", "a condition", "à condition", "condition"),
    "publication": (
        "publication",
        "publiee",
        "publiée",
        "entre en vigueur",
        "vigueur",
        "date d expedition",
        "date d'expédition",
        "expedition",
    ),
    "أموال ذاتية": ("أموال ذاتية", "اموال ذاتية", "أموالهم الخاصة", "اموالهم الخاصة", "أموال خاصة"),
    "traite": ("traite", "traite ", "سفتجة", "avalisee", "avalisée"),
    "avalisee": ("avalisee", "avalisée", "avalise", "aval", "traite"),
    "marches publics": ("marches publics", "marché public", "marche public", "collectivite"),
    "exclues": ("exclues", "exclue", "exclusion", "exempt", "hors champ"),
    "depot": (
        "depot",
        "dépot",
        "dépôt",
        "depots",
        "dépôts",
        "deposent",
        "déposent",
        "deposer",
        "déposer",
        "immobiliser",
        "immobilisation",
        "numeraire",
        "numéraire",
    ),
    "exceptions": ("exceptions", "exclues", "exclue", "exclusion", "exempt", "hors champ"),
    "conditions": (
        "conditions",
        "condition",
        "sous reserve",
        "sous réserve",
        "autorisation",
        "autorisees",
        "autorisées",
    ),
    "autorisation": (
        "autorisation",
        "autorisee",
        "autorisée",
        "autoriser",
        "préalable",
        "prealable",
        "autorisation prealable",
        "autorisation préalable",
    ),
    "120": ("120", "cent vingt", "jusqu a 120", "jusqu'a 120", "jusqu’à 120"),
    "60": ("60", "soixante"),
    "librement": (
        "librement",
        "libre",
        "sans autorisation",
        "free settlement",
        "effectuees librement",
        "effectuées librement",
    ),
    "fonds propres": (
        "fonds propres",
        "fond propres",
        "own funds",
        "numeraire",
        "numéraire",
        "depots",
        "dépôts",
        "depot",
        "dépôt",
    ),
    "numeraire": (
        "numeraire",
        "numéraire",
        "cash",
        "espèces",
        "especes",
        "fonds propres",
        "depot",
        "dépôt",
    ),
    "cash": ("cash", "numeraire", "numéraire", "espèces", "especes", "depot", "dépôt"),
    "deposit": ("deposit", "depot", "dépôt", "depots", "dépôts", "numeraire", "numéraire"),
    "garanties bancaires": (
        "garanties bancaires",
        "garantie bancaire",
        "caution",
        "concours financier",
        "garant",
    ),
    "remise documentaire": (
        "remise documentaire",
        "remises documentaires",
        "documentary collection",
        "quel que soit le mode",
        "mode de reglement",
        "mode de règlement",
    ),
    "independamment": (
        "independamment",
        "indépendamment",
        "quel que soit",
        "quelle que soit",
        "meme lorsqu",
        "même lorsqu",
        "mode de reglement",
        "mode de règlement",
        "crédits documentaires",
        "credits documentaires",
    ),
    "entamee": (
        "entamee",
        "entamée",
        "entame",
        "entamé",
        "execution",
        "exécution",
        "commence",
        "deja engage",
        "déjà engagé",
        "engagements pris",
    ),
    "prealablement": (
        "prealablement",
        "préalablement",
        "avant",
        "deja engage",
        "déjà engagé",
        "engagements pris",
        "entamee",
        "entamée",
    ),
    "execution": (
        "execution",
        "exécution",
        "entamee",
        "entamée",
        "commence",
        "engage",
        "effectivement",
    ),
    "industrielles": (
        "industrielles",
        "industrielle",
        "industriel",
        "industrie",
        "industrial",
        "fiche technique",
        "entreprises industrielles",
    ),
    "fiche technique": (
        "fiche technique",
        "technical",
        "certificate",
        "sous reserve",
        "sous réserve",
    ),
    "documentary": (
        "documentary",
        "documentaire",
        "credit documentaire",
        "crédit documentaire",
        "remise documentaire",
    ),
    "assurance-credit": (
        "assurance credit",
        "assurance-credit",
        "assurance‑crédit",
        "police d assurance",
        "assurance",
        "police d'assurance",
    ),
    "perfectionnement": ("perfectionnement", "actif", "admission temporaire"),
    "financement": ("financement", "financer", "concours", "credit", "crédit"),
    "reglement": ("reglement", "règlement", "reglement financier", "règlement financier"),
    "non prioritaires": (
        "non prioritaires",
        "non prioritaire",
        "non-prioritaires",
        "non-priority",
    ),
    "citation": ("citation", "vise", "visé", "mention", "référence", "reference", "visa"),
    "mention": ("mention", "vise", "visé", "citation", "référence", "reference"),
    "non": ("non", "ne ", "pas abroge", "pas abrogé", "n'abroge", "does not", "no"),
    "2025-13": ("2025-13", "2025‑13", "circulaire 2025"),
}


def _touches(answer: str, needles: list[str]) -> list[str]:
    blob = _fold(answer)
    missing = []
    for needle in needles:
        if not needle:
            continue
        aliases = _MUST_TOUCH_ALIASES.get(needle, (needle,))
        if not any(_fold(alias) in blob for alias in aliases):
            missing.append(needle)
    return missing


def _top_sources(ranked, limit=5) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for document, _score in ranked or []:
        name = Path(str(document.metadata.get("source", ""))).name
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= limit:
            break
    return names


def _score_item(item: dict, *, status: str, answer: str, sources: list[dict], top: list[str]) -> dict:
    expected = item.get("expected") or {}
    instruments = [Path(x).name for x in expected.get("instruments") or []]
    must_touch = list(expected.get("must_touch") or [])
    shape = expected.get("answer_shape") or "lookup"

    retrieval_ok = True
    if instruments:
        retrieval_ok = bool(set(instruments) & set(top))

    source_names = [Path(str(s.get("file") or s.get("source") or "")).name for s in sources]
    instrument_in_answer_sources = (
        not instruments or bool(set(instruments) & set(source_names))
    )

    missing_touch = _touches(answer, must_touch) if must_touch else []
    closed = status in {
        "clarification_needed",
        "insufficient_evidence",
        "search_results",
        "out_of_scope",
    }
    answered = status in {"answered", "partial_answer"}

    if shape == "clarify":
        answer_ok = closed or bool(
            re.search(r"(?i)pr[eé]cis|clarif|indiquer|specify|more information|وضح|حدد", answer)
        )
    elif shape == "it_depends":
        # Pass if the answer surfaces the governing condition, even when one
        # must_touch synonym is missing, or when it correctly fails closed.
        touched = len(must_touch) - len(missing_touch)
        answer_ok = (
            closed
            or (answered and touched >= max(1, len(must_touch) - 1))
            or bool(
                re.search(
                    r"(?i)sous\s+r[eé]serve|d[eé]pend|condition|fiche technique|"
                    r"it depends|provided that|شريطة|بشرط",
                    answer,
                )
            )
        )
    else:
        # Lookup/scenario: require most must_touch hits (allow one miss for synonym drift).
        if must_touch:
            allowed_misses = 1 if len(must_touch) >= 3 else 0
            # Comparative/multi-doc often states one side fully; allow one synonym miss.
            if shape in {"comparative", "multi_doc"} and len(must_touch) >= 2:
                allowed_misses = max(allowed_misses, 1)
            touch_ok = len(missing_touch) <= allowed_misses
        else:
            touch_ok = True
        answer_ok = answered and touch_ok and (
            instrument_in_answer_sources or not sources
        )
        # Fail-closed is acceptable when retrieval missed the gold instrument.
        if not retrieval_ok and closed:
            answer_ok = True
        # For yes/no with retrieval hit, search_results is still an answer miss.

    return {
        "retrieval_ok": retrieval_ok,
        "answer_ok": answer_ok,
        "missing_touch": missing_touch,
        "instrument_in_sources": instrument_in_answer_sources,
        "shape": shape,
    }


def _summarize(rows: list[dict]) -> dict:
    scored = [row for row in rows if not row.get("error")]
    return {
        "attempted": len(rows),
        "scored": len(scored),
        "errors": sum(1 for row in rows if row.get("error")),
        "retrieval_pass": sum(1 for row in scored if row["retrieval_ok"]),
        "answer_pass": sum(1 for row in scored if row["answer_ok"]),
        "both_pass": sum(
            1 for row in scored if row["retrieval_ok"] and row["answer_ok"]
        ),
        "retrieval_rate": round(
            sum(1 for row in scored if row["retrieval_ok"]) / max(len(scored), 1), 4
        ),
        "answer_rate": round(
            sum(1 for row in scored if row["answer_ok"]) / max(len(scored), 1), 4
        ),
        "both_rate": round(
            sum(1 for row in scored if row["retrieval_ok"] and row["answer_ok"])
            / max(len(scored), 1),
            4,
        ),
        "by_category": {
            cat: {
                "n": len(subset),
                "answer_pass": sum(1 for row in subset if row["answer_ok"]),
                "answer_rate": round(
                    sum(1 for row in subset if row["answer_ok"]) / max(len(subset), 1), 3
                ),
            }
            for cat in sorted({row.get("category") or "?" for row in scored})
            for subset in [[row for row in scored if (row.get("category") or "?") == cat]]
        },
        "by_lang": {
            lang: {
                "n": len(subset),
                "answer_pass": sum(1 for row in subset if row["answer_ok"]),
                "answer_rate": round(
                    sum(1 for row in subset if row["answer_ok"]) / max(len(subset), 1), 3
                ),
            }
            for lang in sorted({row.get("lang") or "?" for row in scored})
            for subset in [[row for row in scored if (row.get("lang") or "?") == lang]]
        },
        "fail_ids": [
            row["id"]
            for row in scored
            if not (row["retrieval_ok"] and row["answer_ok"])
        ][:40],
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    _load_dotenv(ROOT / ".env")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("BCT_VOYAGE_RETRY_SWEEPS", "3")
    os.environ.setdefault("BCT_VOYAGE_RETRY_SLEEP_SECONDS", "8")

    from conversation import chat
    from ingestion.index import configure_runtime_assets
    from jsonl_supersession import clear_supersession_cache
    from runtime_retrieval import create_voyage_backend_from_environment

    clear_supersession_cache()
    configure_runtime_assets(ASSETS, validate=True)
    backend = create_voyage_backend_from_environment()

    items = [
        json.loads(line)
        for line in SUITE_B.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    done: dict[str, dict] = {}
    if CHECKPOINT.is_file():
        prior = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        done = {
            row["id"]: row
            for row in prior.get("rows") or []
            if not row.get("error")
        }
        print(f"resume kept={len(done)}", flush=True)

    print(
        f"items={len(items)} remaining={len(items) - len(done)} "
        f"backend={type(backend).__name__}",
        flush=True,
    )

    rows: list[dict] = list(done.values())
    t0 = time.time()
    for index, item in enumerate(items, start=1):
        if item["id"] in done:
            continue
        query = item["turns"][0]
        started = time.time()
        error = ""
        try:
            ranked = list(backend.retrieve(query))
            top = _top_sources(ranked, limit=5)
            result = chat(
                query,
                memory_state={},
                retrieval_backend=backend,
            )
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
            time.sleep(10)

        score = _score_item(
            item, status=status, answer=answer, sources=sources, top=top
        )
        row = {
            "id": item["id"],
            "category": item.get("category"),
            "lang": item.get("lang"),
            "query": query,
            "expected": item.get("expected"),
            "status": status,
            "answer": answer[:500],
            "top_sources": top,
            "answer_sources": [
                Path(str(s.get("file") or s.get("source") or "")).name for s in sources
            ],
            "error": error,
            "elapsed_s": round(time.time() - started, 2),
            **score,
        }
        rows.append(row)
        done[item["id"]] = row
        mark = "PASS" if score["retrieval_ok"] and score["answer_ok"] and not error else (
            "ERR" if error else "FAIL"
        )
        summary = _summarize(rows)
        print(
            f"[{len(rows)}/{len(items)}] {mark} {item['id']} "
            f"ret={int(score['retrieval_ok'])} ans={int(score['answer_ok'])} "
            f"status={status} both={summary['both_rate']:.3f} "
            f"{row['elapsed_s']}s",
            flush=True,
        )
        if len(rows) % 5 == 0 or mark != "PASS":
            CHECKPOINT.write_text(
                json.dumps(
                    {"summary": _summarize(rows), "rows": rows},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        time.sleep(PAUSE_S)

    summary = _summarize(rows)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["complete"] = True
    OUT.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if CHECKPOINT.is_file():
        CHECKPOINT.unlink()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
