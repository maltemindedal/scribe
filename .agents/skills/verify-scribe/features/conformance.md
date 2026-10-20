# Header-card conformance

`verify.sh conformance` regenerates each CI model's header card with `tools/gguf_dump.py` and compares those bytes to the committed file in `crates/gguf/tests/golden/`. It is not part of `verify.sh all`.

## Sub-features

- `dump-stories260K` writes a card for `stories260K.gguf` and matches `stories260K.json`.
- `dump-stories15M` writes a card for `stories15M-q8_0.gguf` and matches `stories15M-q8_0.json`.
- `dump-smollm2` writes a card for `SmolLM2-135M-Instruct-Q8_0.gguf` and matches `SmolLM2-135M-Instruct-Q8_0.json`.
- `dump-qwen3` writes a card for `Qwen3-0.6B-Q8_0.gguf` and matches `Qwen3-0.6B-Q8_0.json`.
- `missing-gguf` prints `SKIP` when `python3` cannot import `gguf` at version `0.19.0`. That skip does not verify the goldens.

## How to get to it (user POV)

- Install `gguf==0.19.0` from `tools/requirements-gguf.txt`.
- Fetch the CI models with `cargo xtask models fetch --set ci`.
- Run `.agents/skills/verify-scribe/scripts/verify.sh conformance`.

## Driving it with verify.sh

Preconditions:

- `verify.sh models` passed, so the four CI GGUF files are on disk.
- `python3` imports `gguf` at version `0.19.0`.

- **Run.** Run `verify.sh conformance`. The output shows `PASS stories260K matches crates/gguf/tests/golden/stories260K.json`, and the same `PASS` line for `stories15M-q8_0`, `SmolLM2-135M-Instruct-Q8_0`, and `Qwen3-0.6B-Q8_0`, followed by `RESULT: PASS (conformance)`.
- **Proof.** `conformance-<stem>.stdout`, `conformance-<stem>.exit`, and the temp file `cmp` read, in the artifacts directory. Each exit file is `0`.
- **Skip.** On a machine where `python3` cannot import `gguf==0.19.0`, the output shows `SKIP conformance: python3 cannot import gguf at version 0.19.0` and `RESULT: PASS with SKIP (conformance): skipped features are unverified`. `conformance.skip` records the reason. Do not treat that result as a pass for this feature.

## Gotchas

- `verify.sh all` does not run this step.
- The ignored Rust test `card_matches_committed_goldens` reads the committed JSON. It does not run Python. A green `model-test` does not replace this comparison.
- A missing model is `FAIL`, not `SKIP`.
- The dumper writes `PATH.tmp` in the golden's directory only when you pass `--out` yourself. `verify.sh conformance` writes its temp file under the artifacts directory and does not replace the committed golden.
