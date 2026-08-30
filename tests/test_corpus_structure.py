from hashlib import sha256
import json

import pytest

from regulatory_graph.corpus_structure import (
    CorpusCacheError,
    ObservedEdition,
    build_structural_bundle,
    inventory_corpus_cache,
    plan_structural_sync,
    resolve_instrument_identity,
)
from regulatory_graph.fixtures import circular_2016_03_fr_bundle
from regulatory_graph.models import GraphChunk
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
    assert resolved.rule == "legacy_cb_ci_filename_v1"
    assert "2017-08" in resolved.corroborating_text
    assert resolved.corroborating_text == resolved.corroborating_text.rstrip()
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
    assert bundle.source_editions[0].identity_verification_status == "VERIFIED"
    assert bundle.source_editions[0].identity_evidence == "filename_identity"
    assert bundle.source_editions[0].identity_rule == "normal_filename_v1"
    assert bundle.source_editions[0].identity_evidence_text is None
    assert bundle.pages[0].source_sha256 == sha256(pdf.read_bytes()).hexdigest().upper()
    assert bundle.chunks[0].page_numbers == (1,)
    assert bundle.provisions == bundle.provision_versions == ()

    pdf.write_bytes(b"changed-pdf")
    with pytest.raises(CorpusCacheError, match="PDF hash mismatch"):
        inventory_corpus_cache(documents, manifest, chunks)


def _structural_fixture():
    fixture = circular_2016_03_fr_bundle()
    edition = fixture.source_editions[0].model_copy(
        update={
            "relative_path": fixture.source_editions[0].filename,
            "extraction_artifact_hash": "b" * 64,
        }
    )
    chunk = GraphChunk(
        uid=f"chunk:{edition.uid}:2:0:{'c' * 16}",
        page_uid=fixture.pages[0].uid,
        chunk_index=0,
        text="Structural test chunk",
        content_hash="c" * 64,
        source_sha256=edition.sha256,
        extraction_artifact_hash="b" * 64,
        extraction_method="native",
        page_numbers=(2,),
    )
    return fixture.model_copy(
        update={
            "source_editions": (edition,),
            "chunks": (chunk,),
            "provisions": (),
            "provision_versions": (),
            "evidence_spans": (),
            "change_events": (),
        }
    )


def test_structural_sync_skips_an_exact_complete_edition():
    bundle = _structural_fixture()
    edition = bundle.source_editions[0]
    observed = ObservedEdition(
        uid=edition.uid,
        logical_edition_uid=edition.uid,
        relative_path=edition.relative_path,
        sha256=edition.sha256,
        extraction_artifact_hash=edition.extraction_artifact_hash,
        page_uids=frozenset(page.uid for page in bundle.pages),
        chunk_uids=frozenset(chunk.uid for chunk in bundle.chunks),
        page_fingerprints=frozenset(
            (
                page.uid,
                page.source_sha256,
                page.extraction_artifact_hash,
                page.text_hash,
            )
            for page in bundle.pages
        ),
        chunk_fingerprints=frozenset(
            (
                chunk.uid,
                chunk.source_sha256,
                chunk.extraction_artifact_hash,
                chunk.content_hash,
            )
            for chunk in bundle.chunks
        ),
        lifecycle_status=edition.lifecycle_status,
        identity_verification_status=edition.identity_verification_status,
        identity_evidence=edition.identity_evidence,
        identity_rule=edition.identity_rule,
        identity_evidence_text=edition.identity_evidence_text,
    )

    plan = plan_structural_sync(bundle, (observed,))

    assert plan.bundle_to_write.source_editions == ()
    assert plan.skipped_edition_uids == (edition.uid,)
    assert plan.repaired_edition_uids == ()
    assert plan.candidate_edition_uids == ()


def test_structural_sync_retains_prior_version_and_scopes_changed_pdf_candidate():
    bundle = _structural_fixture()
    incoming = bundle.source_editions[0]
    prior = ObservedEdition(
        uid=incoming.uid,
        logical_edition_uid=incoming.uid,
        relative_path=incoming.relative_path,
        sha256="d" * 64,
        extraction_artifact_hash="e" * 64,
        page_uids=frozenset({page.uid for page in bundle.pages}),
        chunk_uids=frozenset({chunk.uid for chunk in bundle.chunks}),
        page_fingerprints=frozenset(),
        chunk_fingerprints=frozenset(),
        lifecycle_status="VALIDATED",
        identity_verification_status="VERIFIED",
        identity_evidence="source_fixture",
        identity_rule="fixture_rule",
        identity_evidence_text="fixture evidence",
    )

    plan = plan_structural_sync(bundle, (prior,))

    expected_suffix = incoming.sha256[:16].lower()
    candidate = plan.bundle_to_write.source_editions[0]
    assert candidate.uid == f"{incoming.uid}:{expected_suffix}"
    assert candidate.logical_edition_uid == incoming.uid
    assert candidate.lifecycle_status == "CANDIDATE"
    assert prior.uid not in {
        item.uid for item in plan.bundle_to_write.source_editions
    }
    assert plan.candidate_edition_uids == (candidate.uid,)
    assert plan.skipped_edition_uids == ()
    assert all(page.source_edition_uid == candidate.uid for page in plan.bundle_to_write.pages)
    assert all(candidate.uid.removeprefix("edition:") in page.uid for page in plan.bundle_to_write.pages)
    assert all(chunk.page_uid in {page.uid for page in plan.bundle_to_write.pages} for chunk in plan.bundle_to_write.chunks)


def test_structural_sync_repairs_matching_uids_with_corrupt_graph_hashes():
    bundle = _structural_fixture()
    edition = bundle.source_editions[0]
    observed = ObservedEdition(
        uid=edition.uid,
        logical_edition_uid=edition.uid,
        relative_path=edition.relative_path,
        sha256=edition.sha256,
        extraction_artifact_hash=edition.extraction_artifact_hash,
        page_uids=frozenset(page.uid for page in bundle.pages),
        chunk_uids=frozenset(chunk.uid for chunk in bundle.chunks),
        page_fingerprints=frozenset(
            (page.uid, page.source_sha256, page.extraction_artifact_hash, "0" * 64)
            for page in bundle.pages
        ),
        chunk_fingerprints=frozenset(
            (chunk.uid, chunk.source_sha256, chunk.extraction_artifact_hash, "0" * 64)
            for chunk in bundle.chunks
        ),
        lifecycle_status="VALIDATED",
        identity_verification_status=edition.identity_verification_status,
        identity_evidence=edition.identity_evidence,
        identity_rule=edition.identity_rule,
        identity_evidence_text=edition.identity_evidence_text,
    )

    plan = plan_structural_sync(bundle, (observed,))

    assert plan.repaired_edition_uids == (edition.uid,)
    assert plan.bundle_to_write.pages == bundle.pages
    assert plan.bundle_to_write.chunks == bundle.chunks
