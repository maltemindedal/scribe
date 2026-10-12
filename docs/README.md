# Scribe documentation

Scribe is in early development: the workspace, CI and the GGUF parser exist,
and the rest is planned on the [roadmap](https://github.com/maltemindedal/scribe/issues/1).
These docs cover only what is built.

## Tutorial

| Document | Covers | For |
|---|---|---|
| [Getting started](getting-started.md) | From a fresh machine to a built workspace, passing tests and a running `scribe` binary. | Newcomers |

## How-to guides

| Document | Covers | For |
|---|---|---|
| [Running the model tests](guides/running-the-model-tests.md) | Downloading Qwen3-0.6B-Q8_0 and running the ignored tests against it. | Contributors |

## Reference

| Document | Covers | For |
|---|---|---|
| [CLI](reference/cli.md) | Options, output and exit codes of the `scribe` binary. | Users |
| [Configuration](reference/configuration.md) | Environment variables, the pinned toolchain and workspace settings. | Contributors |
| [GGUF parser](reference/gguf-parser.md) | The `gguf` crate's API, validation rules and error kinds. | Contributors and library users |

## Explanation

| Document | Covers | For |
|---|---|---|
| [Architecture overview](architecture/overview.md) | The nine crates, the boundaries enforced in code, and how the GGUF parser works. | Contributors |
| [ADR 0001: Rust 1.99 as the minimum toolchain](architecture/decisions/0001-rust-1-99-toolchain-floor.md) | Why the workspace requires Rust 1.99. | Contributors |
| [ADR 0002: CI checks correctness only](architecture/decisions/0002-correctness-only-ci.md) | Why CI never measures performance and what runs where. | Contributors |

## Contributing

| Document | Covers | For |
|---|---|---|
| [Contributing](contributing.md) | Development setup, the checks CI runs, issue labels and pull request conventions. | Contributors |
