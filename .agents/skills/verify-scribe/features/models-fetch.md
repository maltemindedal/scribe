# Fetch test models

`cargo xtask models fetch` downloads the GGUF files pinned in `crates/xtask/models.toml` and verifies each file's size and SHA-256. Running it again is a no-op.

## Sub-features

- `fetch-ci` fetches the four-file `ci` set (about 810 MB).
- `fetch-all` adds the large reference models (`--set all`).
- `fetch-idempotent` reports `ok <file> (already present)` on a re-run.

## How to get to it (user POV)

- Run `cargo xtask models fetch [--set ci|all]`. Models go to `models/` or `$SCRIBE_MODELS`.

## Driving it with verify.sh

Preconditions:

- `curl` 7.71 or newer, network access to huggingface.co, and about 1 GB free (much more for `all`).

- **Fetch.** Run `verify.sh models`. Every model line in `models-fetch.stdout` starts with `checking`, `fetched` or `ok`, followed by `RESULT: PASS (models)`.
- **Idempotence.** Run `verify.sh models` again. Every model line now reads `ok ... (already present)`.
- **Side effect.** `models-ls.txt` lists each file with the size from `models.toml`.
- **all set.** Set `SCRIBE_VERIFY_SET=all` and run `verify.sh models`.

## Gotchas

- An existing file that fails its size check or its hash check is left in place. Delete it and re-run. A failed download does not leave a file behind.
- Set `SCRIBE_MODELS=/tmp/...` to test a cold download without touching the shared `models/` cache.
