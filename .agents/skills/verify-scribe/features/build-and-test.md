# Build and test the workspace

A contributor builds every crate and runs the unit tests with the pinned toolchain, as the README quick start and CI do.

## Sub-features

- `build` runs `cargo build --workspace` and exits 0.
- `test` runs `cargo test --workspace`, where every `test result:` line is `ok`.

## How to get to it (user POV)

- Run `cargo build --workspace` and `cargo test --workspace` at the repo root.

## Driving it with verify.sh

Preconditions:

- `verify.sh doctor` passes.

- **Build.** Run `verify.sh build`. The output shows `[build] exit=0`.
- **Test.** Run `verify.sh test`. The output shows `PASS all <N> test result lines are ok` and a count such as `N passed, 7 ignored`.
- **Proof.** `test.stdout` and `test.summary` in the artifacts directory.

## Gotchas

- CI also runs `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` with `RUSTFLAGS=-D warnings`. A local pass here does not cover those checks.
- Exactly seven tests are ignored on purpose. Six need the CI GGUF files under `models/` or `$SCRIBE_MODELS`. One needs llama.cpp reference dumps (see `features/llama-cpp-reference.md`).
