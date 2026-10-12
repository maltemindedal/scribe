# Real-model parser test

The ignored test `parses_qwen3_0_6b_q8_0` parses Qwen3-0.6B-Q8_0 and checks its tensor count, metadata count, vocabulary size and tensor shapes.

## Sub-features

- `parse-qwen3` runs `parses_qwen3_0_6b_q8_0`, which reports `ok`.

## How to get to it (user POV)

- Run `cargo test -p gguf -- --ignored` (see `docs/guides/running-the-model-tests.md`).

## Driving it with verify.sh

Preconditions:

- `verify.sh models` passed, so `Qwen3-0.6B-Q8_0.gguf` is in the models directory.

- **Run.** Run `verify.sh model-test`. The output shows `PASS parses_qwen3_0_6b_q8_0 passes`.
- **Proof.** `model-test.stdout` in the artifacts directory.

## Gotchas

- Without the model the test fails rather than skips. Run `models` first.
- `SCRIBE_MODELS` must be the same for both `models` and `model-test`.
