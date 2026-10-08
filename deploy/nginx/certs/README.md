Put the server's HTTPS certificate here, with exactly these names, for the bundled Nginx
(`docker compose --profile nginx up -d`):

- `bct.crt` — the certificate (with its chain, if the bank's IT gives one)
- `bct.key` — its private key

Without them the bundled Nginx serves plain HTTP. These files are never committed (see `.gitignore`).
