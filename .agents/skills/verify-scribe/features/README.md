# Scribe verification map

This directory is the maintained source for verifying the user-facing behavior of Scribe. Read this index before driving anything, then use the matching feature file as the recipe.

## Baseline preconditions

- A Scribe checkout whose toolchain resolves to Rust 1.99 (`rustup toolchain install` reads `rust-toolchain.toml`).
- `verify.sh doctor` exits 0.
- For model features, `curl` 7.71 or newer and network access to huggingface.co.

## Driving conventions

- Run every step through `.agents/skills/verify-scribe/scripts/verify.sh`.
- Use one `SCRIBE_VERIFY_RUN` id per verification so all evidence lands in one directory.
- Treat commands and expected strings as literal.

## Proof and skip reporting

- CLI proof is the command, stdout, stderr and exit code. `verify.sh` saves all four.
- Report a `SKIP` as unverified, with its reason. Never count it as covered by another path.

## Feature entry contract

Each feature file has an H1 title and a one-paragraph description, then exactly four H2 sections: `Sub-features`, `How to get to it (user POV)`, `Driving it with verify.sh`, `Gotchas`.

## Features

- [Build and test the workspace](./build-and-test.md) covers `cargo build` and `cargo test --workspace`.
- [scribe CLI](./cli.md) covers `--version`, `--help`, no arguments, and unknown commands.
- [Fetch test models](./models-fetch.md) covers `cargo xtask models fetch` and the hash check.
- [Real-model parser test](./model-test.md) covers the ignored `parses_qwen3_0_6b_q8_0` test and the resident-set check on `GgufFile::open`.
- [llama.cpp reference dumps](./llama-cpp-reference.md) covers `tools/llama.cpp/run.sh` and the `bench` KLD/NMSE comparator.
- [OpenAI-compatible server](./openai-server.md) is a stub until #36 lands.
