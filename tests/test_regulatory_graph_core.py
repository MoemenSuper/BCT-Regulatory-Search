from datetime import date

import pytest
from pydantic import ValidationError

from regulatory_graph.models import (
    ChangeEvent,
    EvidenceSpan,
    Instrument,
    InstrumentKind,
    LegalAction,
    ProvisionVersion,
    SourceEdition,
    SourceStatus,
    TargetSelectorType,
    TargetSpan,
    TargetScope,
    VerificationStatus,
    VersionStatus,
)
from regulatory_graph.schema import (
    CORE_NODE_LABELS,
    CORE_RELATIONSHIP_PATTERNS,
    install_schema,
    is_allowed_relationship,
    schema_statements,
)
from regulatory_graph.validation import GraphReferenceError, validate_change_event_for_write


def _verified_event(**overrides) -> ChangeEvent:
    values = {
        "uid": "event:replace:2016-03:article-4",
        "source_instrument_uid": "BCT:CIRCULAR:2016:03",
        "action": LegalAction.REPLACE,
        "target_scope": TargetScope.ARTICLE,
        "target_provision_uids": ("BCT:CIRCULAR:1991:24:ARTICLE:4",),
        "raw_effect_text": "L'article 4 est remplacé par les dispositions suivantes.",
        "effective_from": date(2016, 4, 1),
        "evidence_uids": ("evidence:cir-2016-03:p2:change-1",),
        "confidence": 0.99,
        "verification_status": VerificationStatus.VERIFIED,
    }
    values.update(overrides)
    return ChangeEvent(**values)


def test_verified_change_event_requires_a_target_and_evidence():
    with pytest.raises(ValidationError, match="target"):
        _verified_event(target_provision_uids=())

    with pytest.raises(ValidationError, match="evidence"):
        _verified_event(evidence_uids=())


def test_verified_event_with_unresolved_trigger_cannot_materialize_temporal_state():
    event = _verified_event(
        effective_from=None,
        effective_trigger="NOTIFICATION",
    )

    assert event.temporal_state_ready is False
    assert _verified_event().temporal_state_ready is True


def test_resolved_trigger_requires_explicit_resolution_state():
    unresolved = _verified_event(
        effective_trigger="NOTIFICATION",
        effective_trigger_resolved=False,
    )
    resolved = _verified_event(
        effective_trigger="NOTIFICATION",
        effective_trigger_resolved=True,
    )

    assert unresolved.temporal_state_ready is False
    assert resolved.temporal_state_ready is True


def test_candidate_event_never_materializes_temporal_state():
    event = _verified_event(
        verification_status=VerificationStatus.CANDIDATE,
    )

    assert event.temporal_state_ready is False


def test_provision_version_enforces_half_open_interval_ordering():
    with pytest.raises(ValidationError, match="valid_to"):
        ProvisionVersion(
            uid="version:article-4:v2",
            provision_uid="BCT:CIRCULAR:1991:24:ARTICLE:4",
            version_number=2,
            text="Version consolidée de l'article 4.",
            language="fr",
            valid_from=date(2024, 1, 1),
            valid_to=date(2024, 1, 1),
            status=VersionStatus.SUPERSEDED,
            content_hash="a" * 64,
        )


def test_active_provision_version_must_be_verified():
    with pytest.raises(ValidationError, match="ACTIVE"):
        ProvisionVersion(
            uid="version:article-4:v2",
            provision_uid="BCT:CIRCULAR:1991:24:ARTICLE:4",
            version_number=2,
            text="Version consolidée de l'article 4.",
            language="fr",
            valid_from=date(2024, 1, 1),
            status=VersionStatus.ACTIVE,
            content_hash="a" * 64,
            verification_status=VerificationStatus.CANDIDATE,
        )


def test_active_provision_version_requires_a_resolved_start_date():
    with pytest.raises(ValidationError, match="valid_from"):
        ProvisionVersion(
            uid="version:article-4:v2",
            provision_uid="BCT:CIRCULAR:1991:24:ARTICLE:4",
            version_number=2,
            text="Version consolidee de l'article 4.",
            language="fr",
            status=VersionStatus.ACTIVE,
            content_hash="a" * 64,
            verification_status=VerificationStatus.VERIFIED,
        )


def test_official_language_editions_share_one_canonical_instrument():
    instrument = Instrument(
        uid="BCT:CIRCULAR:2016:03",
        authority="BCT",
        kind=InstrumentKind.CIRCULAR,
        year=2016,
        number="03",
        corpus_present=True,
        source_status=SourceStatus.LOCAL,
    )

    french = SourceEdition(
        uid="edition:cir-2016-03:fr",
        instrument_uid=instrument.uid,
        language="fr",
        filename="Cir_2016_03_fr.pdf",
        sha256="a" * 64,
        extraction_status="complete",
        page_count=3,
        is_scan=False,
    )
    arabic = SourceEdition(
        uid="edition:cir-2016-03:ar",
        instrument_uid=instrument.uid,
        language="ar",
        filename="Cir_2016_03_ar.pdf",
        sha256="b" * 64,
        extraction_status="complete",
        page_count=3,
        is_scan=False,
    )

    assert french.instrument_uid == arabic.instrument_uid == instrument.uid
    assert french.uid != arabic.uid


def test_graph_contracts_are_immutable():
    event = _verified_event()

    with pytest.raises(ValidationError, match="frozen"):
        event.action = LegalAction.ABROGATE


def test_target_scope_requires_a_compatible_target_kind():
    with pytest.raises(ValidationError, match="INSTRUMENT"):
        _verified_event(target_scope=TargetScope.INSTRUMENT)

    with pytest.raises(ValidationError, match="TargetSpan"):
        _verified_event(target_scope=TargetScope.PHRASE)

    event = _verified_event(
        target_scope=TargetScope.INSTRUMENT,
        target_instrument_uids=("BCT:CIRCULAR:1991:24",),
        target_provision_uids=(),
    )

    assert event.target_instrument_uids == ("BCT:CIRCULAR:1991:24",)

    with pytest.raises(ValidationError, match="ARTICLE"):
        _verified_event(
            target_scope=TargetScope.ARTICLE,
            target_provision_uids=(),
            target_span_uids=("target:article-4:sentence-1",),
        )


def test_target_span_has_a_dedicated_sub_provision_selector_vocabulary():
    span = TargetSpan(
        uid="target:cir-2019-10:annex-1:bullet-2",
        provision_version_uid="version:annex-1:v1",
        selector_type=TargetSelectorType.BULLET,
        raw_selector="deuxième tiret",
    )

    assert span.selector_type == TargetSelectorType.BULLET


def test_paragraph_and_item_targets_can_be_provisions_or_nested_spans():
    paragraph = _verified_event(
        target_scope=TargetScope.PARAGRAPH,
        target_provision_uids=(),
        target_span_uids=("target:article-4:paragraph-2",),
    )
    item = _verified_event(
        target_scope=TargetScope.ITEM,
        target_provision_uids=(),
        target_span_uids=("target:annex-1:item-a",),
    )

    assert paragraph.target_span_uids == ("target:article-4:paragraph-2",)
    assert item.target_span_uids == ("target:annex-1:item-a",)


def test_evidence_span_keeps_exact_page_and_extraction_provenance():
    evidence = EvidenceSpan(
        uid="evidence:cir-2016-03:p2:change-1",
        source_edition_uid="edition:cir-2016-03:fr",
        quote="L'article 4 est remplacé.",
        page_number=2,
        extraction_method="native",
        source_sha256="b" * 64,
        extraction_artifact_hash="c" * 64,
        char_start=120,
        char_end=150,
    )

    assert evidence.page_number == 2
    assert evidence.char_end > evidence.char_start


def test_core_relationship_registry_is_precise_and_contains_change_lineage():
    assert len(CORE_RELATIONSHIP_PATTERNS) == len(set(CORE_RELATIONSHIP_PATTERNS))
    assert ("Instrument", "DECLARES_CHANGE", "ChangeEvent") in CORE_RELATIONSHIP_PATTERNS
    assert ("ChangeEvent", "TARGETS", "Provision") in CORE_RELATIONSHIP_PATTERNS
    assert (
        "ChangeEvent",
        "INTRODUCES_VERSION",
        "ProvisionVersion",
    ) in CORE_RELATIONSHIP_PATTERNS
    assert (
        "ProvisionVersion",
        "SUPERSEDES_VERSION",
        "ProvisionVersion",
    ) in CORE_RELATIONSHIP_PATTERNS
    assert all(pattern[1] != "RELATED_TO" for pattern in CORE_RELATIONSHIP_PATTERNS)
    assert is_allowed_relationship("ChangeEvent", "TARGETS", "Provision") is True
    assert is_allowed_relationship("Instrument", "RELATED_TO", "Instrument") is False


def test_verified_change_references_must_resolve_before_graph_write():
    class ReferenceCatalog:
        def __init__(self, missing=()):
            self.missing = set(missing)

        def exists(self, label, uid):
            return (label, uid) not in self.missing

    event = _verified_event()

    assert validate_change_event_for_write(event, ReferenceCatalog()) is event

    with pytest.raises(GraphReferenceError, match="EvidenceSpan"):
        validate_change_event_for_write(
            event,
            ReferenceCatalog({("EvidenceSpan", event.evidence_uids[0])}),
        )


def test_schema_statements_are_idempotent_and_cover_every_core_uid():
    statements = schema_statements()
    constraint_statements = [
        statement for statement in statements if "CREATE CONSTRAINT" in statement
    ]

    assert len(constraint_statements) == len(CORE_NODE_LABELS)
    assert all("IF NOT EXISTS" in statement for statement in statements)
    for label in CORE_NODE_LABELS:
        assert any(f"(node:{label})" in statement for statement in constraint_statements)
        assert any("REQUIRE node.uid IS UNIQUE" in statement for statement in constraint_statements)


def test_schema_installer_uses_an_injected_driver_compatible_seam():
    class RecordingDriver:
        def __init__(self):
            self.calls = []

        def execute_query(self, statement, *, database_):
            self.calls.append((statement, database_))

    driver = RecordingDriver()

    installed = install_schema(driver, database="bct-regulatory-test")

    assert installed == schema_statements()
    assert driver.calls == [
        (statement, "bct-regulatory-test") for statement in schema_statements()
    ]
