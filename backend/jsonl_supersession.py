"""JSONL SUPERSEDES: public seam for ingest extract and retrieve-time pin.

Implementation: supersession_edges (IO/extract/merge) and supersession_pin (pin backend).
"""
from supersession_edges import (  # noqa: F401
    SupersessionEdge,
    edge_key,
    edge_quote_is_operative,
    extract_edges_from_page_text,
    extract_edges_from_pages,
    instrument_from_filename,
    instruments_from_text,
    load_edges,
    load_prior_edges,
    merge_edge_lists,
    merge_supersession_edges_for_ingest,
    rebuild_supersession_edges_from_documents,
    resolve_edges_path,
    write_edges,
    _pdf_page_texts,
)
from supersession_pin import (  # noqa: F401
    SupersessionPinBackend,
    build_page_lookup_from_native,
    clear_supersession_cache,
    maybe_wrap_backend,
    pin_supersession_edges,
    select_edges,
    select_edges_from_hits,
)
