# Persistent local Neo4j

This deployment is the non-disposable local store for the temporal regulatory
graph. Its binary database, logs, imports, and backups live outside Git.

Set the resolved root in the current PowerShell session before any Compose
command:

```powershell
$env:BCT_NEO4J_PERSISTENT_ROOT = 'C:\Users\Moemen Super\BCT-Regulatory-Search-local-data\neo4j\temporal-regulatory-v1'
docker compose -f infrastructure/neo4j/compose.yaml up -d
```

The HTTP and Bolt endpoints bind only to loopback on ports `17474` and `17687`.
Authentication is disabled for this loopback-only experimental deployment; do
not expose either port beyond the local machine.

Stop without deleting the container or bind-mounted data:

```powershell
docker compose -f infrastructure/neo4j/compose.yaml stop
```

Never run the disposable tracer cleanup against this deployment. Do not use
`docker compose down --volumes` for this graph.

Create an offline portable dump with the same pinned image:

```powershell
docker compose -f infrastructure/neo4j/compose.yaml stop
docker run --rm --name bct-temporal-regulatory-v1-dump `
  --env NEO4J_AUTH=none `
  --mount "type=bind,source=$env:BCT_NEO4J_PERSISTENT_ROOT/data,target=/data" `
  --mount "type=bind,source=$env:BCT_NEO4J_PERSISTENT_ROOT/backups,target=/backups" `
  neo4j:2026.07.1@sha256:dbc377fb9cd8fe8dabc19d3041b197d5ca0ef8bae514cea175b8df265e5b7a76 `
  neo4j-admin database dump neo4j --to-path=/backups --overwrite-destination=true
docker compose -f infrastructure/neo4j/compose.yaml start
```

Neo4j Community does not support `STOP DATABASE`, so the server must be fully
stopped before the dump helper mounts `/data`.
