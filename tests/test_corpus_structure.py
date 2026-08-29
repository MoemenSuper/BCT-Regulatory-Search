from hashlib import sha256
import json

import pytest

from regulatory_graph.corpus_structure import (
    CorpusCacheError,
    build_structural_bundle,
    inventory_corpus_cache,
    resolve_instrument_identity,
)
from regulatory_graph.models import InstrumentKind, VerificationStatus


def test_normal_filename_resolves_without_language_becoming_identity():
    french = resolve_instrument_identity("Cir_2022_03_fr.pdf", "")
    arabic = resolve_instrument_identity("Cir_2022_03_ar.pdf", "")

    assert french.status == VerificationStatus.VERIFIED
    assert french.instrument_uid == arabic.instrument_uid == "BCT:CIRCULAR:2022:03"
    assert french.kind == InstrumentKind.CIRCULAR
    assert french.language == "fr"
    assert arabic.language == "ar"


def test_legacy_filename_requires_matching_first_page_identity_evidence():
    resolved = resolve_instrument_identity(
        "CB_2017_08_FR.pdf",
        "CIRCULAIRE AUX BANQUES N°2017-08 Objet : contrôle interne",
    )
    unresolved = resolve_instrument_identity(
        "CB_2017_08_FR.pdf",
        "Document bancaire sans numéro visible",
    )

    assert resolved.status == VerificationStatus.VERIFIED
    assert resolved.instrument_uid == "BCT:CIRCULAR:2017:08"
    assert resolved.evidence == "first_page_identity"
    assert unresolved.status == VerificationStatus.NEEDS_REVIEW
    assert unresolved.instrument_uid.startswith("BCT:UNRESOLVED:")


def test_compact_and_note_variants_resolve_only_with_matching_kind_and_number():
    circular = resolve_instrument_identity(
        "Cir202204_fr.pdf",
        "CIRCULAIRE AUX BANQUES N° 2022-04 Objet : conditions de banque",
    )
    note = resolve_instrument_identity(
        "NB-2018_28_1110_fr.pdf",
        "Note aux intermédiaires agréés N ° 2018-28",
    )

    assert circular.instrument_uid == "BCT:CIRCULAR:2022:04"
    assert note.instrument_uid == "BCT:NOTE:2018:28"
    assert circular.status == note.status == VerificationStatus.VERIFIED


def test_cache_inventory_requires_exact_pdf_artifact_and_chunk_page_provenance(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    pdf = documents / "Cir_2022_03_fr.pdf"
    pdf.write_bytes(b"frozen-pdf")
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "filename": pdf.name,
                "language": "fr",
                "document_number": "2022-03",
                "publication_date": None,
                "pages": [
                    {
                        "page_number": 1,
                        "raw_text": "CIRCULAIRE N° 2022-03",
                        "quality_score": 1.0,
                        "extraction_method": "native",
                        "quality_flags": [],
                        "metadata": {},
                        "blocks": [],
                    }
                ],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": {
                    pdf.name: {
                        "source": pdf.name,
                        "sha256": sha256(pdf.read_bytes()).hexdigest(),
                        "artifact": str(artifact),
                        "pages": 1,
                    }
                },
                "errors": [],
                "document_count": 1,
            }
        ),
        encoding="utf-8",
    )
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "page_content": "CIRCULAIRE N° 2022-03",
                "metadata": {"source": pdf.name, "page": 1, "pages": [1]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    inventory = inventory_corpus_cache(documents, manifest, chunks)

    assert inventory.pdf_count == 1
    assert inventory.page_count == 1
    assert inventory.chunk_count == 1
    assert inventory.editions[0].identity.instrument_uid == "BCT:CIRCULAR:2022:03"

    bundle = build_structural_bundle(inventory)
    assert len(bundle.instruments) == 1
    assert len(bundle.source_editions) == 1
    assert len(bundle.pages) == 1
    assert len(bundle.chunks) == 1
    assert bundle.source_editions[0].relative_path == pdf.name
    assert bundle.source_editions[0].uid == "edition:cir-2022-03-fr"
    assert bundle.source_editions[0].extraction_artifact_hash
    assert bundle.pages[0].source_sha256 == sha256(pdf.read_bytes()).hexdigest().upper()
    assert bundle.chunks[0].page_numbers == (1,)
    assert bundle.provisions == bundle.provision_versions == ()

    pdf.write_bytes(b"changed-pdf")
    with pytest.raises(CorpusCacheError, match="PDF hash mismatch"):
        inventory_corpus_cache(documents, manifest, chunks)
