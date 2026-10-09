import re
import threading
import unicodedata
from functools import lru_cache

import snowballstemmer
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# Keyword tokens for French and Arabic: people type "delai" for "délais", "l'agrément" for
# "agréments", and Arabic without harakat or hamza. Measured on the eval set: BM25 page recall
# 82.9% -> 90.4% (Arabic 76.3% -> 86.5%) against lower().split().
_ELISION = re.compile(r"\b(?:l|d|j|m|n|s|t|c|qu|jusqu|lorsqu|puisqu)['’]")
_ARABIC_MARKS = re.compile(r"[ؐ-ًؚ-ٰٟـ]")  # harakat, tatweel
_TOKEN = re.compile(r"[^\W\d_]+|\d+")  # letters and digits apart: "2024-03" -> 2024, 03
_FRENCH = snowballstemmer.stemmer("french")
_ARABIC = snowballstemmer.stemmer("arabic")
_STEM_LOCK = threading.Lock()  # snowballstemmer objects keep per-call state


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    if not word.isalpha():
        return word
    with _STEM_LOCK:
        return (_ARABIC if "؀" <= word[0] <= "ۿ" else _FRENCH).stemWord(word)


def tokenize(text: str) -> list[str]:
    text = _ELISION.sub(" ", unicodedata.normalize("NFKC", text).casefold())
    text = _ARABIC_MARKS.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text).replace("ى", "ي").replace("ة", "ه")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return [_stem(word) for word in _TOKEN.findall(text)]


def searchable_text(text: str, metadata: dict | None) -> str:
    """What every search reads (keyword search, embeddings, reranker): the chunk's context header
    (document, title, section, element, page: ingestion/chunk.py), then its text. Answers quote the
    text only, so the header is never evidence."""
    context = str((metadata or {}).get("context") or "").strip()
    return f"{context}\n{text}" if context else text


def load_documents_from_chroma(vector_store):
    results = vector_store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(results["documents"], results["metadatas"])
    ]


def create_bm25(documents):
    # BM25Okapi divides by corpus size; an empty collection (e.g. no Arabic OCR chunks yet) has no index.
    if not documents:
        return None
    return BM25Okapi([tokenize(searchable_text(document.page_content, document.metadata)) for document in documents])


def retrieve_bm25(query, bm25, documents, k=15):
    if bm25 is None:
        return []
    return bm25.get_top_n(tokenize(query), documents, n=k)
