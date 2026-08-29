from pathlib import Path


COMPOSE_PATH = Path("infrastructure/neo4j/compose.yaml")
PINNED_IMAGE = (
    "neo4j:2026.07.1@"
    "sha256:dbc377fb9cd8fe8dabc19d3041b197d5ca0ef8bae514cea175b8df265e5b7a76"
)


def test_persistent_neo4j_compose_contract_is_local_pinned_and_bind_mounted():
    compose = COMPOSE_PATH.read_text(encoding="utf-8")

    assert f"image: {PINNED_IMAGE}" in compose
    assert '"127.0.0.1:17474:7474"' in compose
    assert '"127.0.0.1:17687:7687"' in compose
    assert "NEO4J_AUTH: none" in compose
    assert compose.count("${BCT_NEO4J_PERSISTENT_ROOT:?") == 4
    for directory in ("data", "logs", "import", "backups"):
        assert f"}}/{directory}:/{directory}" in compose
    assert "DISPOSABLE" not in compose
