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
| [llama.cpp reference dumps](../tools/llama.cpp/README.md) | Building llama.cpp at the pinned commit, dumping activations and logits, and comparing them by NMSE and KL divergence. | Contributors |
| [Running the model tests](guides/running-the-model-tests.md) | Downloading the pinned test models with `cargo xtask models fetch` and running the ignored tests against them. | Contributors |

## Reference

| Document | Covers | For |
|---|---|---|
| [CLI](reference/cli.md) | Options, output and exit codes of the `scribe` binary. | Users |
| [Configuration](reference/configuration.md) | Environment variables, the pinned toolchain and workspace settings. | Contributors |
| [GGUF parser](reference/gguf-parser.md) | The `gguf` crate's API, validation rules and error kinds. | Contributors and library users |

## Explanation

| Document | Covers | For |
|---|---|---|
| [Architecture overview](architecture/overview.md) | The ten crates, the boundaries enforced in code, and how the GGUF parser works. | Contributors |
| [ADR 0001: Rust 1.99 as the minimum toolchain](architecture/decisions/0001-rust-1-99-toolchain-floor.md) | Why the workspace requires Rust 1.99. | Contributors |
| [ADR 0002: CI checks correctness only](architecture/decisions/0002-correctness-only-ci.md) | Why CI never measures performance and what runs where. | Contributors |

## Models

| Document | Covers | For |
|---|---|---|
| [`MODELS.md`](../MODELS.md) | The pinned test models: licence, role, size, what each exercises, and how to bump a pin. | Contributors |

## Research

| Document | Covers | For |
|---|---|---|
| [`research/README.md`](../research/README.md) | The imported research artifacts (report, `gguf_check.rs`, the `qk` kernels crate), their known caveats, and the tokenizer oracle in `tools/`. Not part of the build. | Contributors |

## Contributing

These live at the repository root, where GitHub looks for them.

| Document | Covers | For |
|---|---|---|
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Development setup, the checks CI runs, test layout, conventions, issue labels and pull requests. | Contributors |
| [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) | Expected behaviour in the project and how to report a violation. | Everyone |
| [`SECURITY.md`](../SECURITY.md) | How to report a vulnerability privately, and what is in scope. | Security researchers |
| [`AGENTS.md`](../AGENTS.md) | Commands, conventions and doc rules for LLM coding agents working in this repo. | Coding agents |
