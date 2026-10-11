"""Relations between circulars ("2017-08 abroge et remplace 2013-15"): the single entry point
other files import from.

- supersession_edges.py: finds these sentences in the PDFs at upload, and reads and writes the
  list of relations (supersession_edges.jsonl, one relation per line, next to the index).
- supersession_pin.py: at search time, brings the page that declares a relation into the
  results when the question is about a circular that was replaced.
A relation proves "X replaced Y"; it never proves that X is still in force today.
"""
from extras.supersession_edges import (  # noqa: F401
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
    resolve_edges_path,
    write_edges,
)
from extras.supersession_pin import (  # noqa: F401
    SupersessionPinBackend,
    build_page_lookup_from_native,
    clear_supersession_cache,
    maybe_wrap_backend,
    pin_supersession_edges,
    label_successor_hits,
    select_edges,
)
