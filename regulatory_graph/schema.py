import re
from typing import Any, Protocol


CORE_NODE_LABELS = (
    "Instrument",
    "SourceEdition",
    "Page",
    "Chunk",
    "Provision",
    "ProvisionVersion",
    "ChangeEvent",
    "TargetSpan",
    "EvidenceSpan",
)


CORE_RELATIONSHIP_PATTERNS = (
    ("Instrument", "HAS_EDITION", "SourceEdition"),
    ("SourceEdition", "HAS_PAGE", "Page"),
    ("Page", "HAS_CHUNK", "Chunk"),
    ("Chunk", "NEXT_CHUNK", "Chunk"),
    ("Instrument", "HAS_PROVISION", "Provision"),
    ("Provision", "CONTAINS_PROVISION", "Provision"),
    ("Provision", "HAS_VERSION", "ProvisionVersion"),
    ("Provision", "CURRENT_VERSION", "ProvisionVersion"),
    ("ProvisionVersion", "SUPERSEDES_VERSION", "ProvisionVersion"),
    ("Instrument", "DECLARES_CHANGE", "ChangeEvent"),
    ("ChangeEvent", "TARGETS", "Instrument"),
    ("ChangeEvent", "TARGETS", "Provision"),
    ("ChangeEvent", "TARGETS", "TargetSpan"),
    ("ChangeEvent", "RETIRES_VERSION", "ProvisionVersion"),
    ("ChangeEvent", "INTRODUCES_VERSION", "ProvisionVersion"),
    ("ChangeEvent", "EVIDENCED_BY", "EvidenceSpan"),
    ("TargetSpan", "WITHIN", "ProvisionVersion"),
    ("EvidenceSpan", "ON_PAGE", "Page"),
    ("EvidenceSpan", "IN_CHUNK", "Chunk"),
)


RANGE_INDEXES = (
    ("Instrument", "kind"),
    ("Instrument", "year"),
    ("Instrument", "number"),
    ("Provision", "provision_type"),
    ("ProvisionVersion", "status"),
    ("ProvisionVersion", "valid_from"),
    ("ProvisionVersion", "valid_to"),
    ("ProvisionVersion", "language"),
    ("ChangeEvent", "action"),
    ("ChangeEvent", "effective_from"),
    ("ChangeEvent", "verification_status"),
)


class QueryExecutor(Protocol):
    def execute_query(self, statement: str, *, database_: str) -> Any: ...


def schema_statements() -> tuple[str, ...]:
    constraints = tuple(
        "CREATE CONSTRAINT "
        f"{_snake_case(label)}_uid_unique IF NOT EXISTS "
        f"FOR (node:{label}) REQUIRE node.uid IS UNIQUE"
        for label in CORE_NODE_LABELS
    )
    indexes = tuple(
        "CREATE INDEX "
        f"{_snake_case(label)}_{property_name}_idx IF NOT EXISTS "
        f"FOR (node:{label}) ON (node.{property_name})"
        for label, property_name in RANGE_INDEXES
    )
    return constraints + indexes


def install_schema(
    driver: QueryExecutor,
    *,
    database: str,
) -> tuple[str, ...]:
    statements = schema_statements()
    for statement in statements:
        driver.execute_query(statement, database_=database)
    return statements


def is_allowed_relationship(
    source_label: str,
    relationship_type: str,
    target_label: str,
) -> bool:
    return (source_label, relationship_type, target_label) in CORE_RELATIONSHIP_PATTERNS


def _snake_case(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
