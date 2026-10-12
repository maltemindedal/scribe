# OpenAI-compatible server

Planned in #36. An HTTP server will expose `POST /v1/chat/completions` (JSON and SSE streaming), `GET /v1/models` and `GET /health`. None of it exists yet. `crates/server` is an empty crate and `scribe` has no `serve` command.

## Sub-features

- `server-health` will cover `GET /health`, which returns 503 while loading and 200 `{"status":"ok"}` when ready. Not implemented.
- `server-models` will cover `GET /v1/models`. Not implemented.
- `server-chat` will cover `POST /v1/chat/completions` as JSON. Not implemented.
- `server-stream` will cover the SSE stream ending in `data: [DONE]`. Not implemented.

## How to get to it (user POV)

- Not reachable today. Expect something like `scribe serve --model <gguf> --port <port>` once #36 lands.

## Driving it with verify.sh

Preconditions:

- None today.

- **Stub.** Run `verify.sh server`. It prints `SKIP server: not implemented yet (issue #36)` and writes `server.skip`.
- **When #36 lands.** The stub prints `FAIL server code exists now`. Replace it with these steps: launch on a free port (pass `--port`, never a fixed shared one), write the PID to `$ART/server.pid` so `cleanup` kills only that process, poll `/health` until it returns 200, then `curl` each route and save the response bodies.

## Gotchas

- A `SKIP` here means unverified. Never report the server as working from the CLI or test steps.
