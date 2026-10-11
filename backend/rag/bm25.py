"""Keyword search (BM25): finds chunks that share words with the question, weighting rare words
more than common ones. It catches what meaning search misses: exact numbers, circular numbers,
article labels, technical terms.

The hard part is the words themselves: people type "delai" for "délais", "l'agrément" for
"agréments", and Arabic without vowel marks or hamza. tokenize() reduces every text to the same
simple form, for the chunks and for the question alike, so these variants still match.
"""
import re
import threading
import unicodedata
from functools import lru_cache

import snowballstemmer
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# French elisions ("l'", "d'", "qu'"...) are cut off so "l'agrément" becomes "agrément".
_ELISION = re.compile(r"\b(?:l|d|j|m|n|s|t|c|qu|jusqu|lorsqu|puisqu)['’]")
_ARABIC_MARKS = re.compile(r"[ؐ-ًؚ-ٰٟـ]")  # harakat, tatweel
_TOKEN = re.compile(r"[^\W\d_]+|\d+")  # letters and digits apart: "2024-03" -> 2024, 03
# Stemmers cut words down to a common root, so "délai" and "délais" count as the same word.
_FRENCH = snowballstemmer.stemmer("french")
_ARABIC = snowballstemmer.stemmer("arabic")
_STEM_LOCK = threading.Lock()  # snowballstemmer objects keep per-call state


@lru_cache(maxsize=200_000)  # the same words come back constantly: remember their root
def _stem(word: str) -> str:
    if not word.isalpha():
        return word  # numbers stay as they are
    with _STEM_LOCK:
        # The first letter tells the language: Arabic letters live in the U+0600-U+06FF block.
        return (_ARABIC if "؀" <= word[0] <= "ۿ" else _FRENCH).stemWord(word)


def tokenize(text: str) -> list[str]:
    """Text -> list of simplified word roots, used for the chunks and the question alike."""
    # Lower case (casefold also handles special cases like "ß"), then drop French elisions.
    text = _ELISION.sub(" ", unicodedata.normalize("NFKC", text).casefold())
    text = _ARABIC_MARKS.sub("", text)
    # Arabic letters that people write interchangeably become one form (all alef variants -> ا,
    # final ى -> ي, ة -> ه), so spelling habits do not change the match.
    text = re.sub("[إأآٱ]", "ا", text).replace("ى", "ي").replace("ة", "ه")
    # Remove French accents: "délai" -> "delai" (split each letter from its accent, drop the accent).
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return [_stem(word) for word in _TOKEN.findall(text)]


def searchable_text(text: str, metadata: dict | None) -> str:
    """What every search reads (keyword search, embeddings, reranker): the chunk's context header
    (document, title, section, element, page: ingestion/chunk.py), then its text. Answers quote the
    text only, so the header is never evidence."""
    context = str((metadata or {}).get("context") or "").strip()
    return f"{context}\n{text}" if context else text


def load_documents_from_chroma(vector_store):
    """Every chunk stored in a Chroma collection, as Documents (BM25 is built from these)."""
    results = vector_store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(results["documents"], results["metadatas"])
    ]


def create_bm25(documents):
    """The keyword index over these chunks, built in memory at startup (None if there are none)."""
    # BM25Okapi divides by corpus size; an empty collection (e.g. no Arabic OCR chunks yet) has no index.
    if not documents:
        return None
    return BM25Okapi([tokenize(searchable_text(document.page_content, document.metadata)) for document in documents])


def retrieve_bm25(query, bm25, documents, k=15):
    """The k chunks sharing the most (rare) words with the question."""
    if bm25 is None:
        return []
    return bm25.get_top_n(tokenize(query), documents, n=k)
